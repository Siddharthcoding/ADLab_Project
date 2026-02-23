# ml/skill_gap_scorer.py
"""
SkillGapScorer + NeuralRoadmapReranker
========================================
Combines forgetting-curve retention, topic embeddings, contest stress signals,
and a simulated-annealing local search to produce the optimal problem order.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.skill_trajectory import (
    TemporalSkillProfile,
    ForgettingCurveEstimator,
    TopicEmbedder,
    TagTimeline,
)


# ---------------------------------------------------------------------------
# 4.  Skill Gap Scorer
# ---------------------------------------------------------------------------

@dataclass
class ProblemScore:
    problem_id: str
    name: str
    source: str
    difficulty: int
    tags: List[str]
    link: str

    # Raw scores (all in [0, 1])
    forgetting_urgency: float = 0.0    # how much the tag has been forgotten
    novelty_score: float = 0.0         # structural gap from known skills
    contest_relevance: float = 0.0     # alignment with contest-critical topics
    difficulty_fit: float = 0.0        # how well difficulty matches user level

    # Final composite
    priority: float = 0.0
    matches_weak_topic: bool = False
    explanation: str = ""              # human-readable reason


CONTEST_CRITICAL = {
    'dp', 'dynamic programming', 'graph', 'graphs', 'tree', 'trees',
    'two pointers', 'binary search', 'dfs', 'bfs', 'greedy',
    'backtracking', 'sliding window', 'heap', 'priority queue',
    'segment tree', 'bit manipulation', 'number theory', 'string'
}


class SkillGapScorer:
    """
    Produces a ProblemScore for each candidate problem by combining:

      priority = w1·forgetting_urgency
               + w2·novelty_score
               + w3·contest_relevance
               + w4·difficulty_fit
               + w5·weak_topic_bonus

    Weights are adjusted based on the user's profile.
    """

    def __init__(
        self,
        profile: TemporalSkillProfile,
        embedder: TopicEmbedder,
        user_level: str = "Intermediate",
        contest_penalty: float = 0.5,
        weak_topics: Optional[List[str]] = None,
    ):
        self.profile = profile
        self.embedder = embedder
        self.weak_set = set(str(t).lower().strip() for t in (weak_topics or []))
        self.user_level = user_level
        self.contest_penalty = contest_penalty

        # User's "known" tag cluster (tags with ≥ 3 recent solves)
        self.known_tags = [
            tag for tag, tl in profile.timelines.items()
            if len(tl.events) >= 3
        ]

        # Infer approximate Elo from user level
        self._user_elo = {
            "Absolute Beginner": 800,
            "Beginner": 950,
            "Intermediate": 1200,
            "Advanced": 1500,
            "Expert": 1800,
        }.get(user_level, 1200)

        # Dynamic weights: experts care more about contest, beginners about forgetting
        if user_level in ("Expert", "Advanced"):
            self._w = dict(forgetting=0.15, novelty=0.20, contest=0.30,
                           diff_fit=0.20, weak=0.15)
        elif user_level == "Intermediate":
            self._w = dict(forgetting=0.30, novelty=0.20, contest=0.20,
                           diff_fit=0.15, weak=0.15)
        else:  # Beginner
            self._w = dict(forgetting=0.35, novelty=0.15, contest=0.10,
                           diff_fit=0.25, weak=0.15)

    # ---- Component scorers ------------------------------------------------

    def _forgetting_urgency(self, tags: List[str]) -> float:
        """
        Maximum forgetting urgency across the problem's tags.
        If a tag was never seen → moderate urgency (0.5).
        """
        scores = []
        for tag in tags:
            tl: Optional[TagTimeline] = self.profile.timelines.get(tag)
            if tl is None:
                scores.append(0.5)   # never practiced → moderate urgency
            else:
                # High retention = low urgency. Low retention = high urgency.
                urgency = 1.0 - tl.retention
                # Boost if last seen very recently (user is actively working on it)
                if tl.last_seen_days < 3:
                    urgency *= 0.6   # de-prioritise; already fresh
                scores.append(float(np.clip(urgency, 0.0, 1.0)))

        return max(scores) if scores else 0.5

    def _novelty_score(self, tags: List[str]) -> float:
        """Average structural gap score for all tags on this problem."""
        if not self.known_tags:
            return 0.5
        scores = [
            self.embedder.cluster_gap_score(self.known_tags, tag)
            for tag in tags
        ]
        return float(np.mean(scores)) if scores else 0.5

    def _contest_relevance(self, tags: List[str]) -> float:
        """Fraction of this problem's tags that are contest-critical."""
        if not tags:
            return 0.0
        overlap = sum(1 for t in tags if t in CONTEST_CRITICAL)
        base = overlap / len(tags)
        # Amplify for users with high contest penalty
        return float(np.clip(base * (1 + self.contest_penalty), 0.0, 1.0))

    def _difficulty_fit(self, difficulty: int) -> float:
        """
        Gaussian bell-curve fit centred at user's stretch zone (Elo + 100).
        Wider sigma for higher-level users so the score doesn't collapse to
        near-zero for problems a few hundred points below the target.
        """
        target = self._user_elo + 100
        # Wider sigma for higher levels: Experts can benefit from 800-rated
        # weak-topic problems too, but harder ones should still score higher.
        sigma = 250 if self._user_elo < 1200 else 450
        score = math.exp(-((difficulty - target) ** 2) / (2 * sigma ** 2))
        return float(score)

    # ---- Main scorer -------------------------------------------------------

    def score(self, problem: dict) -> ProblemScore:
        tags = [str(t).lower().strip() for t in problem.get("tags", [])]
        diff = int(problem.get("difficulty") or 1200)
        pid = str(problem.get("id", ""))

        ps = ProblemScore(
            problem_id=pid,
            name=problem.get("name", "Unknown"),
            source=problem.get("source", ""),
            difficulty=diff,
            tags=tags,
            link=problem.get("link", ""),
            matches_weak_topic=bool(self.weak_set.intersection(set(tags))),
        )

        ps.forgetting_urgency = self._forgetting_urgency(tags)
        ps.novelty_score = self._novelty_score(tags)
        ps.contest_relevance = self._contest_relevance(tags)
        ps.difficulty_fit = self._difficulty_fit(diff)

        weak_bonus = 1.0 if ps.matches_weak_topic else 0.0

        ps.priority = (
            self._w["forgetting"] * ps.forgetting_urgency
            + self._w["novelty"] * ps.novelty_score
            + self._w["contest"] * ps.contest_relevance
            + self._w["diff_fit"] * ps.difficulty_fit
            + self._w["weak"] * weak_bonus
        )

        # Build rich, specific, varied explanation
        ps.explanation = self._build_explanation(ps, tags, diff)
        return ps

    def _build_explanation(self, ps: ProblemScore, tags: List[str], diff: int) -> str:
        """
        Generate a specific, narrative reason for recommending this problem.
        Varies phrasing based on the dominant signal so no two reasons read alike.
        """
        parts = []

        # ── Forgetting signal: name the specific tag and how long ago ──────
        if ps.forgetting_urgency > 0.60:
            worst_tag, worst_days, worst_r = None, 0.0, 1.0
            for t in tags:
                tl = self.profile.timelines.get(t)
                if tl and tl.retention < worst_r:
                    worst_tag, worst_days, worst_r = t, tl.last_seen_days, tl.retention
            if worst_tag:
                days = int(worst_days)
                pct = int((1 - worst_r) * 100)
                if days < 7:
                    parts.append(f"'{worst_tag}' was recently active — reinforce now")
                elif days < 30:
                    parts.append(f"'{worst_tag}' is fading ({pct}% forgotten, {days}d gap) — review soon")
                elif days < 90:
                    parts.append(f"'{worst_tag}' needs refresh — {pct}% forgotten after {days}d")
                else:
                    parts.append(f"'{worst_tag}' critically at risk — {pct}% forgotten, last seen {days}d ago")

        # ── Novelty signal: name the adjacent cluster ──────────────────────
        if ps.novelty_score > 0.60 and not parts:
            known_neighbors = [
                t for t in tags
                if t in self.profile.timelines and len(self.profile.timelines[t].events) >= 2
            ]
            new_tags = [t for t in tags if t not in self.profile.timelines]

            # Vary the phrase by difficulty tier so same-tag problems read differently
            diff_qualifier = (
                f"elite stretch at {diff}" if diff >= 2000 else
                f"contest-level at {diff}" if diff >= 1500 else
                f"fluency-building at {diff}" if diff >= 1200 else
                f"foundation reps at {diff}"
            )

            if known_neighbors:
                anchor = known_neighbors[0]
                bridge_phrases = [
                    f"bridges '{anchor}' into adjacent territory",
                    f"deepens '{anchor}' with new structural patterns",
                    f"extends '{anchor}' — cross-topic application",
                    f"'{anchor}' variant with different problem framing",
                    f"strengthens '{anchor}' ({diff_qualifier})",
                ]
                parts.append(random.choice(bridge_phrases))
            elif new_tags:
                intro_phrases = [
                    f"introduces '{new_tags[0]}' — a gap in your profile",
                    f"'{new_tags[0]}' — unexplored in your history",
                    f"opens '{new_tags[0]}' ({diff_qualifier})",
                ]
                parts.append(random.choice(intro_phrases))
            else:
                parts.append(f"expands skill cluster ({diff_qualifier})")

        # ── Contest relevance: name which contest tag ──────────────────────
        contest_tags = [t for t in tags if t in CONTEST_CRITICAL]
        if ps.contest_relevance > 0.55 and contest_tags:
            ct = contest_tags[0]
            if self.contest_penalty > 0.55:
                parts.append(f"'{ct}' is a frequent contest bottleneck for you")
            elif self.contest_penalty > 0.35:
                parts.append(f"'{ct}' appears in ~{int(ps.contest_relevance*100)}% of contests")
            else:
                parts.append(f"sharpens '{ct}' for time-pressure scenarios")

        # ── Difficulty fit: specific framing by level ──────────────────────
        if ps.difficulty_fit > 0.75:
            gap = diff - self._user_elo
            if gap > 200:
                parts.append(f"rated {diff} — pushes past your comfort zone")
            elif gap > -100:
                parts.append(f"rated {diff} — sits in your optimal stretch zone")
            else:
                parts.append(f"rated {diff} — solidifies your foundation")

        # ── Weak-topic flag: add context ───────────────────────────────────
        if ps.matches_weak_topic:
            matching_weak = [t for t in tags if t in self.weak_set]
            if matching_weak and not any(matching_weak[0] in p for p in parts):
                parts.append(f"directly targets weak area: '{matching_weak[0]}'")

        if not parts:
            # Fallback: describe the tag mix
            tag_str = " + ".join(tags[:2]) if tags else "mixed topics"
            parts.append(f"well-rounded {tag_str} practice")

        # Join with em-dash for a natural reading flow (max 2 parts)
        return " — ".join(parts[:2])


# ---------------------------------------------------------------------------
# 5.  Neural Roadmap Reranker  (simulated annealing)
# ---------------------------------------------------------------------------

class NeuralRoadmapReranker:
    """
    Given a list of ProblemScore objects, use simulated annealing to find
    an ordering that maximises a curriculum quality objective:

      Q(σ) = Σ_i  priority(σ_i) · scaffold_bonus(σ_i | σ_{<i})
             + α · progression_reward(σ)      ← NEW: difficulty ramp
             - β · tag_burst_penalty(σ)        ← STRENGTHENED: tag diversity
             - γ · platform_clump_penalty(σ)   ← NEW: LC/CF interleaving

    All three structural terms operate on the full sequence so SA can
    discover orderings that trade off local priority for global curriculum quality.
    """

    # Tag-budget: at most this many of the same primary tag in any window of K
    TAG_WINDOW = 8
    TAG_MAX_PER_WINDOW = 2

    def __init__(
        self,
        temperature: float = 2.0,
        cooling: float = 0.990,
        min_temp: float = 0.005,
        max_iter: int = 4000,
        tag_penalty_weight: float = 0.8,      # β — hard tag-burst penalty
        progression_weight: float = 0.4,       # α — reward difficulty ramp
        platform_weight: float = 0.15,         # γ — reward LC/CF alternation
    ):
        self.T0 = temperature
        self.cooling = cooling
        self.T_min = min_temp
        self.max_iter = max_iter
        self.beta = tag_penalty_weight
        self.alpha = progression_weight
        self.gamma = platform_weight
        random.seed(42)
        np.random.seed(42)

    # ── Structural term 1: scaffold bonus ────────────────────────────────────

    def _scaffold_bonus(self, ps: ProblemScore, seen_tags: set) -> float:
        """Reward partial tag overlap with previously seen problems."""
        if not ps.tags:
            return 0.0
        overlap = len(set(ps.tags) & seen_tags) / len(ps.tags)
        return float(1.0 - abs(overlap - 0.45) * 2.2) * 0.25

    # ── Structural term 2: difficulty progression reward ─────────────────────

    def _progression_reward(self, ordering: List[ProblemScore]) -> float:
        """
        Reward orderings where difficulty trends upward across the sequence.
        Uses Kendall's τ correlation between position and difficulty as the signal.
        A perfect ascending ramp scores 1.0; random ordering scores ~0; 
        descending scores negative.
        """
        n = len(ordering)
        if n < 4:
            return 0.0
        diffs = [ps.difficulty for ps in ordering]
        # Count concordant vs discordant pairs (Kendall τ, O(n²) but n≤50)
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = diffs[j] - diffs[i]
                if d > 0:
                    concordant += 1
                elif d < 0:
                    discordant += 1
        total_pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / max(total_pairs, 1)
        return float(tau)   # in [-1, +1]

    # ── Structural term 3: tag burst penalty ─────────────────────────────────

    def _tag_burst_penalty(self, ordering: List[ProblemScore]) -> float:
        """
        Penalise any window of TAG_WINDOW consecutive problems that contains
        more than TAG_MAX_PER_WINDOW problems sharing the same primary tag.
        
        Also penalise any 3-in-a-row with identical full tag-sets (near-dupes).
        """
        n = len(ordering)
        penalty = 0.0
        W, M = self.TAG_WINDOW, self.TAG_MAX_PER_WINDOW

        # Sliding window tag-burst
        for start in range(n - W + 1):
            window = ordering[start: start + W]
            primary_tags = [ps.tags[0] if ps.tags else "__none__" for ps in window]
            from collections import Counter
            counts = Counter(primary_tags)
            for tag, cnt in counts.items():
                if tag != "__none__" and cnt > M:
                    penalty += (cnt - M) * 0.5

        # Near-duplicate penalty: identical tag-sets back-to-back
        for i in range(n - 1):
            set_a = frozenset(ordering[i].tags)
            set_b = frozenset(ordering[i + 1].tags)
            if set_a and set_b and set_a == set_b:
                penalty += 1.5
            elif set_a and set_b and len(set_a & set_b) / max(len(set_a | set_b), 1) > 0.70:
                penalty += 0.5   # high-overlap pair

        return penalty

    # ── Structural term 4: platform interleaving reward ───────────────────────

    def _platform_reward(self, ordering: List[ProblemScore]) -> float:
        """Reward LC/CF alternation; penalise runs of 3+ same platform."""
        reward = 0.0
        for i in range(1, len(ordering)):
            if ordering[i].source != ordering[i - 1].source:
                reward += 0.15   # alternation bonus
            if (i >= 2
                    and ordering[i].source == ordering[i - 1].source == ordering[i - 2].source):
                reward -= 0.25   # clump penalty
        return reward

    # ── Full objective ────────────────────────────────────────────────────────

    def _objective(self, ordering: List[ProblemScore]) -> float:
        seen_tags: set = set()
        priority_sum = 0.0

        for i, ps in enumerate(ordering):
            position_decay = 1.0 / (1 + 0.015 * i)
            scaffold = self._scaffold_bonus(ps, seen_tags)
            priority_sum += (ps.priority + scaffold) * position_decay
            seen_tags.update(ps.tags)

        prog   = self._progression_reward(ordering)
        burst  = self._tag_burst_penalty(ordering)
        plat   = self._platform_reward(ordering)

        return (priority_sum
                + self.alpha * prog
                - self.beta  * burst
                + self.gamma * plat)

    def rerank(self, scored_problems: List[ProblemScore]) -> List[ProblemScore]:
        if len(scored_problems) <= 3:
            return sorted(scored_problems, key=lambda p: -p.priority)

        # Smart initialisation: sort by difficulty tiers, interleaving within each tier
        easy   = sorted([p for p in scored_problems if p.difficulty <= 1000],
                        key=lambda x: -x.priority)
        medium = sorted([p for p in scored_problems if 1000 < p.difficulty <= 1500],
                        key=lambda x: -x.priority)
        hard   = sorted([p for p in scored_problems if p.difficulty > 1500],
                        key=lambda x: -x.priority)

        def _interleave_platforms(pool: List[ProblemScore]) -> List[ProblemScore]:
            lc = [p for p in pool if p.source == "LeetCode"]
            cf = [p for p in pool if p.source != "LeetCode"]
            out, li, ci = [], 0, 0
            while li < len(lc) or ci < len(cf):
                if li < len(lc):
                    out.append(lc[li]); li += 1
                if ci < len(cf):
                    out.append(cf[ci]); ci += 1
            return out

        # Start: easy → medium → hard, each tier interleaved by platform
        current = (
            _interleave_platforms(easy)
            + _interleave_platforms(medium)
            + _interleave_platforms(hard)
        )
        best = current[:]
        current_obj = self._objective(current)
        best_obj = current_obj

        T = self.T0
        n = len(current)

        for iteration in range(self.max_iter):
            if T < self.T_min:
                break

            move_type = random.random()
            i, j = sorted(random.sample(range(n), 2))
            candidate = current[:]

            if move_type < 0.50:
                # Swap two elements
                candidate[i], candidate[j] = candidate[j], candidate[i]
            elif move_type < 0.75:
                # Reverse a short segment (max length 6 to avoid ruining progression)
                seg_len = min(j - i + 1, 6)
                j = i + seg_len - 1
                candidate[i:j + 1] = candidate[i:j + 1][::-1]
            else:
                # Insert: move element i to position j
                elem = candidate.pop(i)
                candidate.insert(j, elem)

            cand_obj = self._objective(candidate)
            delta = cand_obj - current_obj

            if delta > 0 or random.random() < math.exp(delta / T):
                current = candidate
                current_obj = cand_obj
                if current_obj > best_obj:
                    best = current[:]
                    best_obj = current_obj

            T *= self.cooling

        return best