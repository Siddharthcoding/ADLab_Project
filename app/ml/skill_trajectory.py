# ml/skill_trajectory.py
"""
Neural Skill Trajectory Predictor
==================================
Models each user's per-topic learning curve using personalized forgetting
curves (Ebbinghaus + power-law decay), a topic embedding space trained via
co-occurrence, and an online Bayesian update to dynamically re-rank the roadmap.

Architecture:
  1. TemporalSkillProfile    – builds a timeline of solve events per tag
  2. ForgettingCurveEstimator – fits a personalised retention curve per tag
  3. TopicEmbedder            – SVD-based co-occurrence embedding of topics
  4. SkillGapScorer           – combines trajectory + embedding → priority score
  5. NeuralRoadmapReranker    – gradient-free local search to reorder roadmap
"""

from __future__ import annotations

import math
import time
import json
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# 1.  Temporal Skill Profile
# ---------------------------------------------------------------------------

@dataclass
class SolveEvent:
    timestamp: float          # Unix epoch of the solve
    tag: str
    difficulty: int           # 800 / 1200 / 1600 for LC; raw rating for CF
    source: str               # 'lc' | 'cf'
    contest: bool = False     # solved under time pressure?


@dataclass
class TagTimeline:
    tag: str
    events: List[SolveEvent] = field(default_factory=list)

    # derived — populated after .fit()
    retention: float = 1.0          # current estimated retention [0,1]
    decay_rate: float = 0.07        # λ in e^{-λ·Δt}
    learning_rate_est: float = 0.5  # how fast the user learns this tag
    last_seen_days: float = 999.0   # days since last practice

    def add_event(self, ev: SolveEvent):
        self.events.append(ev)
        self.events.sort(key=lambda e: e.timestamp)


class TemporalSkillProfile:
    """
    Builds per-tag timelines from raw submission lists.

    Inputs:
      lc_subs  – list of dicts with keys: titleSlug, statusDisplay, timestamp (optional)
      cf_subs  – list of dicts with keys: problem{contestId,index,tags}, verdict, creationTimeSeconds
      lc_df    – normalized LC DataFrame with problem_id → tags
      cf_df    – normalized CF DataFrame with problem_id → tags
    """

    def __init__(self):
        self.timelines: Dict[str, TagTimeline] = {}
        self._now = time.time()

    def _get_or_create(self, tag: str) -> TagTimeline:
        if tag not in self.timelines:
            self.timelines[tag] = TagTimeline(tag=tag)
        return self.timelines[tag]

    def ingest_cf(self, cf_subs: List[dict], cf_df):
        """Parse CF submissions into solve events."""
        # Build pid → tags index from catalog
        pid_tags: Dict[str, List[str]] = {}
        if cf_df is not None and not cf_df.empty:
            for _, row in cf_df.iterrows():
                pid = str(row.get("problem_id", ""))
                tags = row.get("tags", [])
                if isinstance(tags, list):
                    pid_tags[pid] = [t.lower().strip() for t in tags]

        for sub in cf_subs:
            if sub.get("verdict") != "OK":
                continue
            prob = sub.get("problem", {})
            cid = prob.get("contestId")
            idx = prob.get("index")
            if not cid or not idx:
                continue

            pid = f"{cid}{idx}"
            ts = float(sub.get("creationTimeSeconds", self._now))
            diff = int(prob.get("rating") or 1200)
            in_contest = sub.get("author", {}).get("participantType") == "CONTESTANT"

            tags = pid_tags.get(pid, [t.lower() for t in prob.get("tags", [])])
            for tag in tags:
                ev = SolveEvent(timestamp=ts, tag=tag, difficulty=diff,
                                source="cf", contest=in_contest)
                self._get_or_create(tag).add_event(ev)

    def ingest_lc(self, lc_subs: List[dict], lc_df, lc_total_solved: int = 0):
        """
        Parse LC submissions into solve events.

        Timestamp strategy:
          - If the submission carries a real timestamp, use it.
          - Otherwise (LC GraphQL returns ~20 subs without timestamps), spread
            them evenly across a 3–30 day recency window so the forgetting curve
            sees them as "recently active" rather than all stale on the same day.

        Bayesian prior: lc_total_solved is stored so ForgettingCurveEstimator
        can apply a retention floor for data-sparse tags on high-volume users.
        """
        slug_tags: Dict[str, List[str]] = {}
        slug_diff: Dict[str, int] = {}
        if lc_df is not None and not lc_df.empty:
            for _, row in lc_df.iterrows():
                slug = str(row.get("problem_id", ""))
                tags = row.get("tags", [])
                diff = int(row.get("difficulty") or 1200)
                if isinstance(tags, list):
                    slug_tags[slug] = [t.lower().strip() for t in tags]
                slug_diff[slug] = diff

        accepted = [s for s in lc_subs if s.get("statusDisplay") == "Accepted"]
        n = len(accepted)
        WINDOW_DAYS = 30   # spread no-timestamp subs across this recency window
        MIN_DAYS = 3       # most recent assumed sub is 3 days ago

        for i, sub in enumerate(accepted):
            slug = sub.get("titleSlug", "")
            raw_ts = sub.get("timestamp")
            if raw_ts:
                ts = float(raw_ts)
            else:
                # Spread across window: index 0 = newest, index n-1 = oldest
                fraction = (i / (n - 1)) if n > 1 else 0.0
                days_back = MIN_DAYS + fraction * (WINDOW_DAYS - MIN_DAYS)
                ts = self._now - days_back * 86400

            tags = slug_tags.get(slug, [])
            diff = slug_diff.get(slug, 1200)
            for tag in tags:
                ev = SolveEvent(timestamp=ts, tag=tag, difficulty=diff, source="lc")
                self._get_or_create(tag).add_event(ev)

        # Store as Bayesian prior for the estimator
        self._lc_total_solved = lc_total_solved

    @property
    def all_tags(self) -> List[str]:
        return list(self.timelines.keys())


# ---------------------------------------------------------------------------
# 2.  Forgetting Curve Estimator
# ---------------------------------------------------------------------------

def _ebbinghaus(t: np.ndarray, lam: float, r: float) -> np.ndarray:
    """
    Modified Ebbinghaus: R(t) = exp(-λ · t^r)
    where t is time in days, λ is decay rate, r is power (< 1 = sublinear decay).
    """
    return np.exp(-lam * np.power(t + 1e-9, r))


class ForgettingCurveEstimator:
    """
    For each tag timeline, fit a personalised forgetting curve using the
    inter-solve intervals as implicit retention probes.

    Strategy:
      - Each successful re-solve of a tag after Δt days is treated as
        "evidence of retention at time t" → R(Δt) ≈ 1.
      - Each LONG gap before re-solving is treated as "potential forgetting".
      - Fit _ebbinghaus() to the (Δt, R) pairs using nonlinear least squares.
      - Predict current retention given days-since-last-solve.

    Bayesian prior (Fix for data-sparse tags on expert users):
      If lc_total_solved is high (≥100), a tag with only 1–2 data points has
      almost certainly been practiced extensively on LeetCode — we just can't
      see the timestamps. We apply a retention FLOOR scaled to lc_total_solved
      so that fundamental topics like "two pointers" aren't marked 91% forgotten
      when the user has solved 1160 LC problems.
    """

    DEFAULT_LAMBDA = 0.08
    DEFAULT_R = 0.6

    # Fundamental tags that an Expert almost certainly retains well
    FUNDAMENTAL_TAGS = {
        'array', 'string', 'math', 'two pointers', 'hash table',
        'sorting', 'binary search', 'sliding window', 'greedy',
        'linked list', 'recursion', 'simulation', 'prefix sum',
    }

    def _bayesian_retention_floor(
        self, tag: str, lc_total_solved: int, n_events: int
    ) -> float:
        """
        Return the minimum retention we should believe given the user's
        overall LC volume and how little local evidence we have.

        A user with 1000+ LC solves almost certainly retains 'two pointers'
        even if we only see 1 CF solve 300 days ago.
        """
        if lc_total_solved < 50 or n_events >= 5:
            return 0.0    # Enough data, or low-volume user — no floor needed

        # Scale floor: 100 solves → 0.45 floor, 500+ → 0.80 floor
        volume_factor = min(lc_total_solved / 500.0, 1.0)
        base_floor = 0.40 + 0.40 * volume_factor        # 0.40 … 0.80

        # Fundamental topics get the full floor; niche topics get half
        tag_lower = tag.lower().strip()
        topic_factor = 1.0 if tag_lower in self.FUNDAMENTAL_TAGS else 0.5

        return base_floor * topic_factor

    def fit_timeline(self, tl: TagTimeline, lc_total_solved: int = 0) -> TagTimeline:
        events = tl.events
        now_days = time.time() / 86400

        if len(events) < 2:
            # Not enough data — use population defaults, then apply Bayesian floor
            last_ts = events[-1].timestamp if events else (time.time() - 30 * 86400)
            tl.last_seen_days = (time.time() - last_ts) / 86400
            tl.decay_rate = self.DEFAULT_LAMBDA
            raw_retention = float(_ebbinghaus(
                np.array([tl.last_seen_days]),
                self.DEFAULT_LAMBDA, self.DEFAULT_R
            )[0])
            floor = self._bayesian_retention_floor(tl.tag, lc_total_solved, len(events))
            tl.retention = max(raw_retention, floor)
            return tl

        # Build inter-solve gaps
        timestamps_days = [e.timestamp / 86400 for e in events]
        gaps = np.diff(timestamps_days)  # Δt between consecutive solves

        # Difficulty-weighted retention proxy:
        # Hard problems solved after a gap → strong retention signal
        diff_weights = np.array([
            1.0 + (events[i + 1].difficulty - 800) / 800
            for i in range(len(events) - 1)
        ], dtype=float)
        diff_weights = np.clip(diff_weights, 0.5, 2.5)

        # Retention signal: fast re-solve = R ≈ 1, slow re-solve = R ≈ decay
        # We use: R_observed = exp(- gap / median_gap) as a normalised proxy
        median_gap = max(np.median(gaps), 0.5)
        R_obs = np.exp(-gaps / (median_gap * 3)) * diff_weights
        R_obs = np.clip(R_obs, 0.01, 1.0)

        try:
            popt, _ = curve_fit(
                _ebbinghaus, gaps, R_obs,
                p0=[self.DEFAULT_LAMBDA, self.DEFAULT_R],
                bounds=([0.001, 0.1], [2.0, 1.5]),
                maxfev=2000
            )
            lam, r = float(popt[0]), float(popt[1])
        except Exception:
            lam, r = self.DEFAULT_LAMBDA, self.DEFAULT_R

        tl.decay_rate = lam

        # Estimate learning rate from solve frequency trend
        if len(events) >= 4:
            early_freq = len(events[:len(events) // 2]) / max(
                (timestamps_days[len(events) // 2] - timestamps_days[0]), 1)
            late_freq = len(events[len(events) // 2:]) / max(
                (timestamps_days[-1] - timestamps_days[len(events) // 2]), 1)
            tl.learning_rate_est = float(np.clip(late_freq / max(early_freq, 1e-6), 0.1, 3.0))
        else:
            tl.learning_rate_est = 1.0

        # Current retention prediction
        last_ts_days = timestamps_days[-1]
        tl.last_seen_days = max(now_days - last_ts_days, 0.0)
        raw_retention = float(_ebbinghaus(
            np.array([tl.last_seen_days]), lam, r
        )[0])
        floor = self._bayesian_retention_floor(tl.tag, lc_total_solved, len(events))
        tl.retention = max(raw_retention, floor)

        return tl


# ---------------------------------------------------------------------------
# 3.  Topic Embedder  (SVD on co-occurrence matrix)
# ---------------------------------------------------------------------------

class TopicEmbedder:
    """
    Builds a dense embedding for each topic tag using SVD decomposition
    of a tag co-occurrence matrix constructed from the problem catalogs.

    Two tags are "co-occurring" if they appear on the same problem.
    High co-occurrence → nearby in embedding space → similar skill requirements.

    This lets us find "neighbour topics" that the user hasn't practised but
    are structurally adjacent to their weaknesses.
    """

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.vocab: List[str] = []
        self.vocab_idx: Dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None   # shape (V, n_components)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, lc_df, cf_df):
        """Build vocab and co-occurrence matrix from both catalogs."""
        # Collect all tag lists per problem
        all_tag_sets: List[List[str]] = []

        def _extract(df, tag_col="tags"):
            if df is None or df.empty:
                return
            for _, row in df.iterrows():
                raw = row.get(tag_col, [])
                tags = [str(t).lower().strip() for t in (raw if isinstance(raw, list) else [])]
                if tags:
                    all_tag_sets.append(tags)

        _extract(lc_df)
        _extract(cf_df)

        if not all_tag_sets:
            return

        # Build vocabulary
        freq: Dict[str, int] = defaultdict(int)
        for ts in all_tag_sets:
            for t in ts:
                freq[t] += 1

        # Keep tags appearing in ≥ 3 problems
        self.vocab = sorted(t for t, c in freq.items() if c >= 3)
        self.vocab_idx = {t: i for i, t in enumerate(self.vocab)}
        V = len(self.vocab)

        if V < 4:
            return

        # Co-occurrence matrix (symmetric, weighted by PMI-like score)
        cooc = np.zeros((V, V), dtype=np.float32)
        for ts in all_tag_sets:
            idxs = [self.vocab_idx[t] for t in ts if t in self.vocab_idx]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    cooc[idxs[i], idxs[j]] += 1
                    cooc[idxs[j], idxs[i]] += 1

        # Apply PPMI (Positive Pointwise Mutual Information)
        total = cooc.sum()
        row_sums = cooc.sum(axis=1, keepdims=True) + 1e-9
        col_sums = cooc.sum(axis=0, keepdims=True) + 1e-9
        ppmi = np.log((cooc * total) / (row_sums * col_sums) + 1e-9)
        ppmi = np.maximum(ppmi, 0)

        # SVD decomposition
        n_comp = min(self.n_components, V - 1)
        self._svd.n_components = n_comp
        raw_emb = self._svd.fit_transform(ppmi)
        self.embeddings = normalize(raw_emb)   # L2-normalise rows

    def get_embedding(self, tag: str) -> Optional[np.ndarray]:
        if self.embeddings is None:
            return None
        idx = self.vocab_idx.get(tag.lower().strip())
        if idx is None:
            return None
        return self.embeddings[idx]

    def nearest_topics(self, tag: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return k most similar tags by cosine similarity."""
        emb = self.get_embedding(tag)
        if emb is None or self.embeddings is None:
            return []
        sims = self.embeddings @ emb          # (V,)
        top_k = np.argsort(-sims)[1: k + 1]  # exclude self
        return [(self.vocab[i], float(sims[i])) for i in top_k]

    def cluster_gap_score(self, known_tags: List[str], candidate_tag: str) -> float:
        """
        Score how much `candidate_tag` fills a gap relative to the user's
        known tag cluster. Higher = more novel yet structurally adjacent.
        """
        known_embs = [self.get_embedding(t) for t in known_tags if self.get_embedding(t) is not None]
        cand_emb = self.get_embedding(candidate_tag)
        if not known_embs or cand_emb is None:
            return 0.5

        known_center = normalize(np.mean(known_embs, axis=0, keepdims=True))[0]
        similarity_to_cluster = float(known_center @ cand_emb)  # [−1, 1]
        # We want: close enough to be learnable, far enough to be novel
        # Sweet spot: 0.2 ≤ sim ≤ 0.7
        novelty_bonus = 1.0 - abs(similarity_to_cluster - 0.45) / 0.55
        return float(np.clip(novelty_bonus, 0.0, 1.0))