# ml/pipeline.py
"""
ML Pipeline Orchestrator
=========================
Single entry point that wires together all ML components and produces:
  - Reranked roadmap (list of dicts compatible with main.py)
  - Per-tag retention report
  - Neighbour topic discovery ("You should also try:")
  - Skill trajectory summary for the terminal dashboard
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.skill_trajectory import (
    TemporalSkillProfile,
    ForgettingCurveEstimator,
    TopicEmbedder,
)
from ml.skill_gap_scorer import (
    SkillGapScorer,
    NeuralRoadmapReranker,
    ProblemScore,
    CONTEST_CRITICAL,
)
from ml.knowledge_graph import TopicKnowledgeGraph, RatingTrajectoryPredictor
from ml.session_planner import SessionPlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_level_from_solved(total: int) -> str:
    if total == 0:
        return "Absolute Beginner"
    if total < 25:
        return "Beginner"
    if total < 100:
        return "Intermediate"
    if total < 300:
        return "Advanced"
    return "Expert"


def _roadmap_item_to_dict(ps: ProblemScore, original: dict) -> dict:
    """Convert ProblemScore back to a roadmap-compatible dict."""
    return {
        **original,
        "ml_priority": round(ps.priority, 4),
        "forgetting_urgency": round(ps.forgetting_urgency, 3),
        "novelty_score": round(ps.novelty_score, 3),
        "contest_relevance": round(ps.contest_relevance, 3),
        "difficulty_fit": round(ps.difficulty_fit, 3),
        "matches_weak_topic": ps.matches_weak_topic,
        "ml_explanation": ps.explanation,
    }


# ---------------------------------------------------------------------------
# Retention Report
# ---------------------------------------------------------------------------

class RetentionReport:
    """Summarises per-tag forgetting curves for display."""

    def __init__(self, profile: TemporalSkillProfile, estimator: ForgettingCurveEstimator,
                 lc_total_solved: int = 0):
        self.entries: List[dict] = []
        for tag, tl in profile.timelines.items():
            fitted = estimator.fit_timeline(tl, lc_total_solved=lc_total_solved)
            self.entries.append({
                "tag": tag,
                "retention": round(fitted.retention, 3),
                "last_seen_days": round(fitted.last_seen_days, 1),
                "decay_rate": round(fitted.decay_rate, 4),
                "solves": len(fitted.events),
                "learning_rate": round(fitted.learning_rate_est, 2),
            })
        self.entries.sort(key=lambda e: e["retention"])   # worst first

    @property
    def at_risk(self) -> List[dict]:
        """Tags with retention < 0.4."""
        return [e for e in self.entries if e["retention"] < 0.4]

    @property
    def strong(self) -> List[dict]:
        """Tags with retention > 0.75."""
        return [e for e in self.entries if e["retention"] > 0.75]

    def print_summary(self, top_n: int = 8):
        print("\n" + "=" * 60)
        print("🧠 ML: SKILL RETENTION ANALYSIS (Forgetting Curves)")
        print("=" * 60)

        if not self.entries:
            print("   Not enough temporal data to fit forgetting curves.")
            return

        print(f"\n⚠️  AT-RISK TOPICS (retention < 40%):")
        at_risk = self.at_risk[:top_n]
        if at_risk:
            for e in at_risk:
                bar = _retention_bar(e["retention"])
                print(f"   {bar}  {e['tag']:<28}  "
                      f"R={e['retention']:.0%}  "
                      f"last={e['last_seen_days']:.0f}d ago  "
                      f"solves={e['solves']}")
        else:
            print("   ✅ No topics at critical risk right now.")

        print(f"\n✅ STRONG TOPICS (retention > 75%):")
        strong = self.strong[:top_n]
        if strong:
            for e in reversed(strong[-top_n:]):
                bar = _retention_bar(e["retention"])
                print(f"   {bar}  {e['tag']:<28}  R={e['retention']:.0%}")
        else:
            print("   Not enough practice history for confident assessment.")


def _retention_bar(r: float, width: int = 10) -> str:
    filled = round(r * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Neighbour Discovery
# ---------------------------------------------------------------------------

class NeighbourTopicDiscovery:
    """
    Find topics structurally adjacent to the user's weak spots that they
    have NEVER practised — these are high-value "unlock" topics.
    """

    def __init__(self, embedder: TopicEmbedder, profile: TemporalSkillProfile):
        self.embedder = embedder
        self.profile = profile
        self.practiced = set(profile.timelines.keys())

    def discover(self, weak_topics: List[str], k: int = 5) -> List[dict]:
        candidates: Dict[str, float] = {}

        for wt in weak_topics:
            neighbours = self.embedder.nearest_topics(wt, k=10)
            for neighbour_tag, sim in neighbours:
                if neighbour_tag not in self.practiced:
                    # Bonus if it's contest-critical
                    cc_bonus = 0.2 if neighbour_tag in CONTEST_CRITICAL else 0.0
                    score = sim + cc_bonus
                    candidates[neighbour_tag] = max(candidates.get(neighbour_tag, 0), score)

        # Sort by score
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [
            {"tag": tag, "score": round(score, 3), "contest_critical": tag in CONTEST_CRITICAL}
            for tag, score in sorted_candidates[:k]
        ]

    def print_discoveries(self, discoveries: List[dict]):
        if not discoveries:
            return
        print("\n" + "=" * 60)
        print("🔭 ML: UNDISCOVERED ADJACENT TOPICS")
        print("    (Topics you haven't tried but are structurally close)")
        print("=" * 60)
        for d in discoveries:
            cc = " 🏆 contest-critical" if d["contest_critical"] else ""
            print(f"   • {d['tag']:<30}  relevance={d['score']:.2f}{cc}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class MLPipeline:
    """
    Full ML pipeline. Call `run()` to get enhanced roadmap.

    Usage:
        pipeline = MLPipeline()
        result = pipeline.run(
            lc_subs=lc_subs, cf_subs=cf_subs,
            lc_df=lc_df, cf_df=cf_df,
            roadmap=roadmap,          # from generate_roadmap()
            weak_topics=weak_topics,
            contest_penalty=combined_penalty,
            lc_total_solved=lc_total_solved,
            cf_total_solved=cf_total_solved,
        )
    """

    def __init__(self):
        self.profile = TemporalSkillProfile()
        self.estimator = ForgettingCurveEstimator()
        self.embedder = TopicEmbedder(n_components=32)

        self.retention_report: Optional[RetentionReport] = None
        self.discoveries: List[dict] = []
        self.scored_problems: List[ProblemScore] = []
        self.knowledge_graph: Optional[TopicKnowledgeGraph] = None
        self.gnn_confidences: Dict[str, float] = {}
        self.session_planner: Optional[SessionPlanner] = None
        self._lc_df = None
        self._cf_df = None
        self._weak_topics: List[str] = []
        self._contest_penalty: float = 0.5
        self._enhanced_roadmap: List[dict] = []

    def run(
        self,
        lc_subs: List[dict],
        cf_subs: List[dict],
        lc_df,
        cf_df,
        roadmap: List[dict],
        weak_topics: List[str],
        contest_penalty: float,
        lc_total_solved: int = 0,
        cf_total_solved: int = 0,
    ) -> List[dict]:

        total_solved = lc_total_solved + cf_total_solved
        user_level = _user_level_from_solved(total_solved)

        print("\n" + "=" * 60)
        print("🤖 ML ENGINE: NEURAL SKILL TRAJECTORY ANALYSIS")
        print("=" * 60)

        # ------------------------------------------------------------------
        # Step 1: Build temporal profile
        # ------------------------------------------------------------------
        t0 = time.time()
        print("   [1/5] Building temporal skill profile...")
        self.profile.ingest_cf(cf_subs, cf_df)
        self.profile.ingest_lc(lc_subs, lc_df, lc_total_solved=lc_total_solved)
        n_tags = len(self.profile.timelines)
        print(f"        → Modelled {n_tags} unique topic timelines")

        # ------------------------------------------------------------------
        # Step 2: Fit forgetting curves
        # ------------------------------------------------------------------
        print("   [2/5] Fitting personalised forgetting curves (Ebbinghaus model)...")
        for tag, tl in self.profile.timelines.items():
            self.estimator.fit_timeline(tl, lc_total_solved=lc_total_solved)
        print(f"        → Fitted {n_tags} forgetting curves")

        # ------------------------------------------------------------------
        # Step 3: Build topic embedding space
        # ------------------------------------------------------------------
        print("   [3/5] Building SVD topic embedding space (PPMI co-occurrence)...")
        self.embedder.fit(lc_df, cf_df)
        v = len(self.embedder.vocab)
        print(f"        → Embedded {v} topics into {self.embedder._svd.n_components}D space")

        # ------------------------------------------------------------------
        # Step 4: Score each roadmap problem
        # ------------------------------------------------------------------
        print("   [4/5] Scoring roadmap problems (multi-factor ML priority)...")
        scorer = SkillGapScorer(
            profile=self.profile,
            embedder=self.embedder,
            user_level=user_level,
            contest_penalty=contest_penalty,
            weak_topics=weak_topics,
        )

        self.scored_problems = []
        for problem in roadmap:
            ps = scorer.score(problem)
            self.scored_problems.append((ps, problem))

        # ------------------------------------------------------------------
        # Step 5: Rerank with simulated annealing
        # ------------------------------------------------------------------
        print("   [5/5] Optimising ordering via simulated annealing curriculum...")
        scored_only = [ps for ps, _ in self.scored_problems]
        orig_map = {ps.problem_id: orig for ps, orig in self.scored_problems}

        reranker = NeuralRoadmapReranker(
            temperature=2.0,
            cooling=0.990,
            max_iter=4000,
        )
        reranked = reranker.rerank(scored_only)

        elapsed = time.time() - t0
        print(f"        → Done in {elapsed:.2f}s")

        # ------------------------------------------------------------------
        # Build output + enforce difficulty diversity
        # ------------------------------------------------------------------
        enhanced_roadmap = []
        for ps in reranked:
            orig = orig_map.get(ps.problem_id, {})
            enhanced_roadmap.append(_roadmap_item_to_dict(ps, orig))

        # Post-process: ensure difficulty progression for advanced/expert users.
        # The raw roadmap from generate_roadmap() may be capped at max_diff=1526
        # for a user with contest_penalty≈0.5, meaning no 1600/1800/2000+ problems
        # get through at all. We enforce a minimum tier distribution:
        #   Expert   → at most 30% of items should be difficulty ≤ 1000
        #   Advanced → at most 40% of items should be difficulty ≤ 1000
        if user_level in ("Expert", "Advanced") and lc_df is not None and cf_df is not None:
            enhanced_roadmap = self._enforce_difficulty_distribution(
                enhanced_roadmap, user_level, lc_df, cf_df,
                weak_topics, lc_total_solved
            )

        # Attach ML artefacts
        self.retention_report = RetentionReport(self.profile, self.estimator, lc_total_solved)
        discovery = NeighbourTopicDiscovery(self.embedder, self.profile)
        self.discoveries = discovery.discover(weak_topics, k=6)

        # ── GNN Knowledge Graph ────────────────────────────────────────────────
        self.knowledge_graph = TopicKnowledgeGraph()
        self.knowledge_graph.build(self.profile, self.embedder, lc_df, cf_df)
        self.gnn_confidences = self.knowledge_graph.run_message_passing()
        # Soft-adjust roadmap priorities using GNN confidence
        enhanced_roadmap = self.knowledge_graph.adjust_roadmap_scores(
            enhanced_roadmap, self.gnn_confidences
        )

        # ── Session Planner ────────────────────────────────────────────────────
        try:
            self.session_planner = SessionPlanner(
                profile=self.profile,
                embedder=self.embedder,
                roadmap=enhanced_roadmap,
                weak_topics=weak_topics,
                contest_penalty=contest_penalty,
            )
        except Exception:
            self.session_planner = None

        # Store for print_insights
        self._lc_df = lc_df
        self._cf_df = cf_df
        self._weak_topics = weak_topics
        self._contest_penalty = contest_penalty
        self._enhanced_roadmap = enhanced_roadmap

        return enhanced_roadmap

    def _enforce_difficulty_distribution(
        self,
        roadmap: List[dict],
        user_level: str,
        lc_df,
        cf_df,
        weak_topics: List[str],
        lc_total_solved: int,
    ) -> List[dict]:
        """
        Three-phase fix for Expert/Advanced roadmap quality:

        Phase 1 — Platform balance audit:
          Count existing LC vs CF. If one platform has < 35% of top-50,
          set an injection quota to restore ~45–55% balance.

        Phase 2 — Tiered difficulty injection:
          Split the injection budget into three tiers (not by raw priority-desc sort
          which always grabs the highest-rated problems first):
            • Tier A (transition): 1200–1499  →  25% of budget
            • Tier B (stretch):    1500–1999  →  45% of budget
            • Tier C (elite):      2000–2400  →  30% of budget
          Within each tier, sort by: weak-topic match first, then difficulty ASC
          so problems step up gradually rather than all landing at 2400.

        Phase 3 — Merge with three-way interleave:
          Arrange final list as: easy foundation → tier-A → tier-B → tier-C,
          each section platform-interleaved. Run _break_adjacent_dupes last.
        """
        n = len(roadmap)
        if n == 0:
            return roadmap

        # ── Phase 0: measure current state ───────────────────────────────────
        easy_count  = sum(1 for p in roadmap if int(p.get("difficulty", 1200)) <= 1000)
        lc_count    = sum(1 for p in roadmap if p.get("source") == "LeetCode")
        cf_count    = n - lc_count
        easy_frac   = easy_count / n
        lc_frac     = lc_count / n

        max_easy_frac = 0.30 if user_level == "Expert" else 0.40
        if easy_frac <= max_easy_frac and abs(lc_frac - 0.50) < 0.15:
            return roadmap   # already healthy

        to_replace = max(0, easy_count - int(n * max_easy_frac))

        print(f"        ⚠️  {easy_frac:.0%} easy (target ≤{max_easy_frac:.0%})  |  "
              f"LC={lc_count} CF={cf_count} (target ~50/50)")
        print(f"        🔧 Tiered injection: {to_replace} harder + platform rebalance...")

        # ── Phase 1: mine candidates from catalogs ────────────────────────────
        weak_set  = set(str(t).lower().strip() for t in (weak_topics or []))
        user_elo  = 1800 if user_level == "Expert" else 1500
        in_roadmap = {str(p.get("id", "")) for p in roadmap}

        # Tier boundaries
        TIERS = [
            ("A", 1200, 1499),
            ("B", 1500, 1999),
            ("C", 2000, 2600),
        ]

        def _inject_reason(tags: List[str], diff: int, matches: bool,
                           source: str) -> str:
            gap  = diff - user_elo
            weak_tag = next((t for t in tags if t in weak_set), None)
            tier = ("elite-level"    if gap > 300 else
                    "stretch zone"   if gap > 50  else
                    "target rating"  if gap > -200 else
                    "building fluency")

            if weak_tag and matches:
                verb = random.choice(["closes", "attacks", "sharpens", "exposes gaps in"])
                return f"{verb} '{weak_tag}' — {tier} ({diff})"
            tag_phrases = {
                "dp": ["multi-state DP", "memoisation + optimal substructure",
                       "DP on intervals", "bottom-up DP design"],
                "dynamic programming": ["DP transition design", "DP with bitmask"],
                "graph": ["graph traversal", "shortest path reasoning",
                          "connected components", "topological ordering"],
                "dfs": ["DFS + state tracking", "recursive graph search"],
                "bfs": ["BFS level design", "multi-source BFS"],
                "binary search": ["non-obvious search space", "binary search on answer"],
                "greedy": ["exchange argument greedy", "greedy with sorting",
                           "interval scheduling greedy"],
                "tree": ["tree DFS + state", "LCA pattern", "tree DP"],
                "trees": ["tree decomposition", "heavy-light decomposition"],
                "segment tree": ["range-query data structure", "lazy propagation"],
                "heap": ["priority queue optimisation"],
                "bit manipulation": ["bitwise DP", "XOR properties"],
                "math": ["number theory insight", "modular arithmetic",
                         "combinatorics counting"],
                "number theory": ["prime factorisation", "Euler's theorem application"],
                "string": ["string hashing", "KMP / Z-function", "trie application"],
            }
            for tag in tags:
                if tag in tag_phrases:
                    phrase = random.choice(tag_phrases[tag])
                    return f"{phrase} — {tier} ({diff})"
            return f"{'CF' if source == 'Codeforces' else 'LC'}-style challenge — {tier} ({diff})"

        def _mine(df, source_label: str, diff_col: str = "difficulty") -> List[dict]:
            out = []
            if df is None or df.empty:
                return out
            for _, row in df.iterrows():
                raw_diff = row.get(diff_col) or row.get("rating") or 1200
                diff = int(raw_diff)
                if diff <= 1000:          # only want harder problems
                    continue
                pid = str(row.get("problem_id", ""))
                if pid in in_roadmap:
                    continue
                tags = row.get("tags", [])
                tags = [str(t).lower().strip()
                        for t in (tags if isinstance(tags, list) else [])]
                matches = bool(weak_set.intersection(set(tags)))
                out.append({
                    "source": source_label,
                    "id": pid,
                    "name": (row.get("name") or row.get("question_title") or "Unknown"),
                    "difficulty": diff,
                    "tags": tags,
                    "link": row.get("link", ""),
                    "matches_weak_topic": matches,
                    "ml_priority": 0.6 + (0.2 if matches else 0.0),
                    "ml_explanation": _inject_reason(tags, diff, matches, source_label),
                    "forgetting_urgency": 0.45,
                    "novelty_score": 0.50,
                    "contest_relevance": 0.50,
                    "difficulty_fit": 0.75,
                })
            return out

        lc_pool = _mine(lc_df, "LeetCode")
        cf_pool = _mine(cf_df, "Codeforces")

        # Sort each pool: weak-topic first, then difficulty ASCENDING (not desc!)
        def _sort_key(p):
            return (0 if p["matches_weak_topic"] else 1, p["difficulty"])

        lc_pool.sort(key=_sort_key)
        cf_pool.sort(key=_sort_key)

        # ── Phase 2: build tiered injection with platform balance ─────────────
        # Target platform split after injection
        target_lc = int(n * 0.48)   # ~48% LC
        lc_deficit = max(0, target_lc - lc_count)   # how many extra LC we need
        cf_deficit = max(0, (n - target_lc) - cf_count)

        tier_fracs = {"A": 0.25, "B": 0.45, "C": 0.30}
        injection_lc: List[dict] = []
        injection_cf: List[dict] = []

        for tier_name, lo, hi in TIERS:
            budget = max(1, round(to_replace * tier_fracs[tier_name]))
            # Platform split within this tier: skew toward whichever is deficient
            if lc_deficit > cf_deficit:
                lc_share = min(int(budget * 0.60), lc_deficit)
            elif cf_deficit > lc_deficit:
                lc_share = max(int(budget * 0.40), budget - cf_deficit)
            else:
                lc_share = budget // 2
            cf_share = budget - lc_share

            tier_lc = [p for p in lc_pool if lo <= p["difficulty"] <= hi][:lc_share]
            tier_cf = [p for p in cf_pool if lo <= p["difficulty"] <= hi][:cf_share]

            injection_lc.extend(tier_lc)
            injection_cf.extend(tier_cf)

            used_ids = {p["id"] for p in tier_lc + tier_cf}
            lc_pool = [p for p in lc_pool if p["id"] not in used_ids]
            cf_pool = [p for p in cf_pool if p["id"] not in used_ids]

            lc_deficit = max(0, lc_deficit - len(tier_lc))
            cf_deficit = max(0, cf_deficit - len(tier_cf))

        injection = injection_lc + injection_cf
        if not injection:
            return roadmap

        total_injected = len(injection)
        d_min = min(p["difficulty"] for p in injection)
        d_max = max(p["difficulty"] for p in injection)
        inj_lc = sum(1 for p in injection if p["source"] == "LeetCode")
        inj_cf = total_injected - inj_lc
        print(f"        ✅ Injected {total_injected} problems | "
              f"rated {d_min}–{d_max} | LC={inj_lc} CF={inj_cf}")

        # ── Phase 3: assemble with difficulty sections, then global interleave ─
        # Keep the best easy problems as a warm-up foundation
        foundation_count = int(n * max_easy_frac)
        easy_pool    = [p for p in roadmap if int(p.get("difficulty", 1200)) <= 1000]
        medium_pool  = [p for p in roadmap if int(p.get("difficulty", 1200)) >  1000]

        # Best foundation: weak-topic matches first
        foundation = sorted(easy_pool,
                             key=lambda p: (0 if p.get("matches_weak_topic") else 1)
                             )[:foundation_count]

        # Tier buckets for injected items, sorted ascending in difficulty
        tier_A = sorted([p for p in injection if p["difficulty"] <= 1499],
                         key=lambda p: p["difficulty"])
        tier_B = sorted([p for p in injection if 1500 <= p["difficulty"] <= 1999],
                         key=lambda p: p["difficulty"])
        tier_C = sorted([p for p in injection if p["difficulty"] >= 2000],
                         key=lambda p: p["difficulty"])

        # Concatenate in curriculum order: easy → transition → stretch → elite
        ordered = foundation + medium_pool + tier_A + tier_B + tier_C
        return self._post_process(ordered)[:n]

    @staticmethod
    def _post_process(roadmap: List[dict]) -> List[dict]:
        """
        Unified single-pass post-processor enforcing two constraints:
          1. No ≥3 consecutive same-platform (debt-counter look-ahead swap).
          2. No adjacent pair with ≥70% tag overlap (look-ahead swap, same-platform preferred).
        Running both in one forward sweep prevents them fighting each other.
        """
        out = roadmap[:]
        n   = len(out)
        plat_run = 1

        for i in range(n):
            # ── Platform constraint ───────────────────────────────────────────
            if i > 0:
                plat_run = plat_run + 1 if out[i].get("source") == out[i-1].get("source") else 1
                if plat_run >= 3:
                    want = ("LeetCode" if out[i].get("source") != "LeetCode"
                            else "Codeforces")
                    for k in range(i + 1, min(i + 12, n)):
                        if out[k].get("source") == want:
                            out.insert(i, out.pop(k))
                            plat_run = 1
                            break

            # ── Near-dupe constraint ──────────────────────────────────────────
            if i < n - 1:
                si = frozenset(out[i].get("tags", []))
                sj = frozenset(out[i+1].get("tags", []))
                if si and sj and len(si & sj) / max(len(si | sj), 1) >= 0.70:
                    src_j = out[i+1].get("source", "")
                    for prefer_same in [True, False]:
                        for k in range(i + 2, min(i + 12, n)):
                            sk = frozenset(out[k].get("tags", []))
                            if prefer_same and out[k].get("source") != src_j:
                                continue
                            if sk and len(si & sk) / max(len(si | sk), 1) < 0.50:
                                out.insert(i + 1, out.pop(k))
                                break
                        else:
                            continue
                        break
        return out

    @staticmethod
    def _break_adjacent_dupes(roadmap: List[dict]) -> List[dict]:
        """Kept for compatibility; _post_process supersedes this in Phase 3."""
        return MLPipeline._post_process(roadmap)

    @staticmethod
    def _interleave_by_platform(roadmap: List[dict]) -> List[dict]:
        """Strict LC/CF alternation used within tier assembly."""
        lc = [p for p in roadmap if p.get("source") == "LeetCode"]
        cf = [p for p in roadmap if p.get("source") != "LeetCode"]
        if not lc or not cf:
            return roadmap
        out, li, ci = [], 0, 0
        while li < len(lc) and ci < len(cf):
            out.append(lc[li]); li += 1
            out.append(cf[ci]); ci += 1
        out.extend(lc[li:])
        out.extend(cf[ci:])
        return out

    @staticmethod
    def _global_platform_interleave(roadmap: List[dict]) -> List[dict]:
        """Debt-based interleave (kept for compatibility)."""
        return MLPipeline._post_process(roadmap)

    def print_insights(self, user_elo: int = 1500, session_hours: float = 2.0):
        """Print all ML-generated insights to terminal."""

        # ── 1. Retention report (forgetting curves) ───────────────────────────
        if self.retention_report:
            self.retention_report.print_summary()

        # ── 2. Undiscovered adjacent topics ───────────────────────────────────
        nd = NeighbourTopicDiscovery(self.embedder, self.profile)
        if self.discoveries:
            nd.print_discoveries(self.discoveries)

        # ── 3. Skill trajectory trends ────────────────────────────────────────
        if self.profile.timelines:
            improving = [
                tag for tag, tl in self.profile.timelines.items()
                if tl.learning_rate_est > 1.2
            ]
            declining = [
                tag for tag, tl in self.profile.timelines.items()
                if tl.learning_rate_est < 0.8 and len(tl.events) >= 3
            ]
            if improving or declining:
                print("\n" + "=" * 60)
                print("📈 ML: SKILL TRAJECTORY TRENDS")
                print("=" * 60)
                if improving:
                    print(f"\n   📈 Accelerating improvement:")
                    for t in sorted(improving, key=lambda x: -self.profile.timelines[x].learning_rate_est)[:5]:
                        lr = self.profile.timelines[t].learning_rate_est
                        print(f"      • {t:<30}  growth rate ×{lr:.1f}")
                if declining:
                    print(f"\n   📉 Slowing practice pace:")
                    for t in sorted(declining, key=lambda x: self.profile.timelines[x].learning_rate_est)[:5]:
                        lr = self.profile.timelines[t].learning_rate_est
                        print(f"      • {t:<30}  growth rate ×{lr:.1f}")

        # ── 4. Knowledge graph + GNN hidden gaps + rating trajectory ──────────
        if self.knowledge_graph and self.gnn_confidences:
            hidden_gaps = self.knowledge_graph.detect_hidden_gaps()
            rtp = RatingTrajectoryPredictor()
            self.knowledge_graph.print_graph_report(
                hidden_gaps=hidden_gaps,
                final_conf=self.gnn_confidences,
                rating_predictor=rtp,
                user_elo=user_elo,
            )

        # ── 5. Session planner outputs ────────────────────────────────────────
        if self.session_planner:
            try:
                self.session_planner.print_forecast_report()
                self.session_planner.print_review_schedule()
                self.session_planner.print_avoidance_report()
                self.session_planner.print_session_plan(hours=session_hours)
                self.session_planner.print_contest_calendar()
            except Exception as e:
                pass   # session planner is bonus — never crash the main output