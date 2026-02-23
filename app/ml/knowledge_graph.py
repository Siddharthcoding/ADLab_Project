# ml/knowledge_graph.py
"""
Topic Knowledge Graph + GNN-style Message Passing
===================================================

This module builds a directed prerequisite graph over CP topics,
then runs 2 rounds of GNN-style message passing to propagate
skill confidence across edges — revealing HIDDEN mastery gaps
that per-topic retention scores alone cannot detect.

Why a graph?
  Per-topic retention treats each tag independently.
  But in reality: if your 'greedy' is 12% retained, your ability
  to solve 'dp' problems that require greedy sub-steps is also
  compromised, even if your isolated 'dp' retention is 89%.
  Message passing surfaces this cross-topic degradation.

Graph construction:
  Nodes  = topics in the embedding vocab
  Edges  = directed prerequisite links
           topic_A → topic_B  iff:
             1. cos_sim(emb_A, emb_B) > SIM_THRESHOLD  (structurally related)
             2. avg_difficulty(B) > avg_difficulty(A)   (B is harder)
             3. edge weight = similarity × difficulty_ratio

Message passing (GraphSAGE mean aggregation, 2 rounds):
  h_v^{(0)} = user_skill_state(v)           [retention × learning_rate]
  h_v^{(1)} = ReLU(W1 · MEAN(h_u : u∈N(v)) + b1)
  h_v^{(2)} = sigmoid(W2 · MEAN(h_u : u∈N(v)^{(1)}) + b2)
  final_confidence(v) = 0.5 · h_v^{(0)} + 0.5 · h_v^{(2)}

  W1, W2 are 1×1 (scalar) since we have 1D node features.
  They are analytically set (no training needed):
    W1 = 0.6  (propagate 60% of neighbour state)
    b1 = 0.1  (baseline confidence floor)
    W2 = 0.7
    b2 = 0.05

Hidden gap detection:
  If raw_retention(v) - propagated_confidence(v) > GAP_THRESHOLD:
    → "hidden gap": topic appears strong in isolation but
       its prerequisites are weak, undermining real ability.

Rating trajectory predictor:
  Uses the GNN-adjusted confidence scores + historical solve pace
  to project the user's effective Codeforces-equivalent Elo
  at 7, 30, and 90-day horizons.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SIM_THRESHOLD   = 0.25   # minimum cosine similarity to draw a prerequisite edge
GAP_THRESHOLD   = 0.18   # raw - propagated gap to flag as hidden
W1, B1          = 0.60, 0.10
W2, B2          = 0.70, 0.05

# Manually curated prerequisite direction hints (A is prereq of B)
# These override the difficulty heuristic when known
KNOWN_PREREQS: List[Tuple[str, str]] = [
    ("two pointers",        "sliding window"),
    ("two pointers",        "binary search"),
    ("array",               "two pointers"),
    ("array",               "prefix sum"),
    ("hash table",          "sliding window"),
    ("sorting",             "greedy"),
    ("sortings",            "greedy"),
    ("greedy",              "dp"),
    ("dfs",                 "backtracking"),
    ("dfs",                 "trees"),
    ("bfs",                 "shortest paths"),
    ("graphs",              "shortest paths"),
    ("graph",               "dfs"),
    ("graph",               "bfs"),
    ("math",                "number theory"),
    ("math",                "combinatorics"),
    ("dp",                  "dfs and similar"),
    ("binary search",       "binary indexed tree"),
    ("data structures",     "segment tree"),
    ("implementation",      "constructive algorithms"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    tag: str
    raw_retention: float      # from Ebbinghaus model
    learning_rate: float      # from TagTimeline
    n_solves: int             # event count
    avg_difficulty: float     # average problem difficulty for this tag
    initial_skill: float = 0.0    # h^(0) = retention × learning_rate_capped
    propagated: float = 0.0       # h^(2) after 2-round message passing
    final_confidence: float = 0.0 # blend of raw + propagated
    hidden_gap: bool = False
    hidden_gap_reason: str = ""


@dataclass
class PrereqEdge:
    src: str    # prerequisite topic
    dst: str    # dependent topic
    weight: float


@dataclass
class HiddenGap:
    topic: str
    apparent_retention: float
    true_confidence: float
    gap_magnitude: float
    weak_prerequisites: List[str]
    insight: str


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class TopicKnowledgeGraph:
    """
    Builds and runs inference on the topic prerequisite graph.
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[PrereqEdge] = []
        self._adj_in: Dict[str, List[Tuple[str, float]]] = defaultdict(list)   # dst → [(src, w)]
        self._adj_out: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # src → [(dst, w)]
        self._fitted = False

    # ── Graph construction ────────────────────────────────────────────────────

    def build(
        self,
        profile,           # TemporalSkillProfile
        embedder,          # TopicEmbedder
        lc_df,
        cf_df,
    ):
        self._profile = profile
        """
        1. Create nodes from the user's topic timelines.
        2. Compute average difficulty per tag from catalogs.
        3. Draw edges from embedding similarity + difficulty ordering.
        4. Augment with curated prerequisite hints.
        """
        # ── Step 1: nodes ─────────────────────────────────────────────────────
        avg_diffs = self._compute_avg_difficulties(lc_df, cf_df)

        for tag, tl in profile.timelines.items():
            skill_0 = float(np.clip(
                tl.retention * min(tl.learning_rate_est, 1.5) / 1.5,
                0.0, 1.0
            ))
            self.nodes[tag] = GraphNode(
                tag=tag,
                raw_retention=tl.retention,
                learning_rate=tl.learning_rate_est,
                n_solves=len(tl.events),
                avg_difficulty=avg_diffs.get(tag, 1200.0),
                initial_skill=skill_0,
            )

        # ── Step 2: edges from embedding ──────────────────────────────────────
        if embedder.embeddings is not None and len(self.nodes) >= 2:
            tags_in_graph = list(self.nodes.keys())
            for i, tag_a in enumerate(tags_in_graph):
                emb_a = embedder.get_embedding(tag_a)
                if emb_a is None:
                    continue
                diff_a = self.nodes[tag_a].avg_difficulty

                for tag_b in tags_in_graph[i + 1:]:
                    emb_b = embedder.get_embedding(tag_b)
                    if emb_b is None:
                        continue
                    sim = float(emb_a @ emb_b)
                    if sim < SIM_THRESHOLD:
                        continue

                    diff_b = self.nodes[tag_b].avg_difficulty
                    if abs(diff_a - diff_b) < 50:
                        continue   # too similar in difficulty — not a clear prerequisite

                    # Lower difficulty → prerequisite
                    prereq, dep = (tag_a, tag_b) if diff_a < diff_b else (tag_b, tag_a)
                    diff_ratio = abs(diff_b - diff_a) / max(diff_a, diff_b)
                    weight = float(sim * (0.5 + 0.5 * diff_ratio))
                    self._add_edge(prereq, dep, weight)

        # ── Step 3: curated hints (higher weight) ─────────────────────────────
        for src, dst in KNOWN_PREREQS:
            if src in self.nodes and dst in self.nodes:
                # Remove any existing low-weight edge and replace
                self._adj_in[dst] = [(s, w) for s, w in self._adj_in[dst] if s != src]
                self._adj_out[src] = [(d, w) for d, w in self._adj_out[src] if d != dst]
                self._add_edge(src, dst, weight=0.80)

        self._fitted = True

    def _add_edge(self, src: str, dst: str, weight: float):
        if src == dst:
            return
        # Deduplicate: keep highest weight
        existing = next((w for s, w in self._adj_in[dst] if s == src), None)
        if existing is not None:
            if weight <= existing:
                return
            self._adj_in[dst] = [(s, w) for s, w in self._adj_in[dst] if s != src]
            self._adj_out[src] = [(d, w) for d, w in self._adj_out[src] if d != dst]
        self.edges.append(PrereqEdge(src=src, dst=dst, weight=weight))
        self._adj_in[dst].append((src, weight))
        self._adj_out[src].append((dst, weight))

    @staticmethod
    def _compute_avg_difficulties(lc_df, cf_df) -> Dict[str, float]:
        tag_diffs: Dict[str, List[float]] = defaultdict(list)
        for df in [lc_df, cf_df]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                diff = float(row.get("difficulty") or row.get("rating") or 1200)
                tags = row.get("tags", [])
                if isinstance(tags, list):
                    for t in tags:
                        tag_diffs[t.lower().strip()].append(diff)
        return {t: float(np.mean(ds)) for t, ds in tag_diffs.items() if ds}

    # ── GNN Message Passing ───────────────────────────────────────────────────

    def run_message_passing(self) -> Dict[str, float]:
        """
        GraphSAGE-style mean aggregation, 2 rounds.
        Returns final confidence per tag.
        """
        if not self._fitted or not self.nodes:
            return {}

        h = {tag: node.initial_skill for tag, node in self.nodes.items()}

        for _round, (W, b) in enumerate([(W1, B1), (W2, B2)]):
            h_new = {}
            for tag in self.nodes:
                neighbours_in = self._adj_in.get(tag, [])
                if neighbours_in:
                    # Weighted mean of prerequisite skill states
                    total_w = sum(w for _, w in neighbours_in)
                    agg = sum(h.get(src, 0.5) * w for src, w in neighbours_in) / total_w
                else:
                    agg = h[tag]   # no prerequisites → self-loop

                # Non-linearity
                raw = W * agg + b
                h_new[tag] = float(np.clip(raw, 0.0, 1.0))
            h = h_new

        # Final blend: 50% original skill + 50% propagated
        final = {}
        for tag, node in self.nodes.items():
            conf = 0.50 * node.initial_skill + 0.50 * h.get(tag, node.initial_skill)
            node.propagated = h.get(tag, node.initial_skill)
            node.final_confidence = round(float(conf), 4)
            final[tag] = node.final_confidence

        return final

    # ── Hidden gap detection ──────────────────────────────────────────────────

    def detect_hidden_gaps(self) -> List[HiddenGap]:
        """
        Find topics where raw_retention >> final_confidence.
        This means the topic LOOKS strong in isolation but its
        prerequisite chain is weak — so real solving ability is lower.
        """
        gaps: List[HiddenGap] = []

        for tag, node in self.nodes.items():
            gap = node.raw_retention - node.final_confidence
            if gap < GAP_THRESHOLD:
                continue

            # Identify which prerequisites are dragging the score down
            weak_prereqs = [
                src for src, w in self._adj_in.get(tag, [])
                if src in self.nodes and self.nodes[src].raw_retention < 0.50
            ]

            if not weak_prereqs:
                continue   # gap exists but no clear weak prereq — skip

            insight = self._build_gap_insight(tag, node, weak_prereqs)
            node.hidden_gap = True
            node.hidden_gap_reason = insight

            gaps.append(HiddenGap(
                topic=tag,
                apparent_retention=round(node.raw_retention, 3),
                true_confidence=round(node.final_confidence, 3),
                gap_magnitude=round(gap, 3),
                weak_prerequisites=weak_prereqs[:3],
                insight=insight,
            ))

        return sorted(gaps, key=lambda g: -g.gap_magnitude)

    @staticmethod
    def _build_gap_insight(tag: str, node: GraphNode, weak_prereqs: List[str]) -> str:
        prereq_str = " + ".join(f"'{p}'" for p in weak_prereqs[:2])
        ret_pct    = int(node.raw_retention * 100)
        conf_pct   = int(node.final_confidence * 100)
        return (
            f"'{tag}' shows {ret_pct}% retention in isolation, but "
            f"weak prerequisites ({prereq_str}) drag real confidence to ~{conf_pct}%. "
            f"Fixing prerequisites will unlock faster '{tag}' improvement."
        )

    # ── Roadmap re-scoring with GNN confidence ────────────────────────────────

    def adjust_roadmap_scores(
        self, roadmap: List[dict], final_conf: Dict[str, float]
    ) -> List[dict]:
        """
        Multiply each problem's ml_priority by the GNN confidence of its
        weakest prerequisite tag.  Problems whose prerequisite chain is weak
        get slightly deprioritised in favour of fixing foundations first.
        """
        for p in roadmap:
            tags = [t.lower() for t in p.get("tags", [])]
            if not tags:
                continue

            # Find the minimum GNN confidence among this problem's tags
            min_conf = min(
                (final_conf.get(t, 0.5) for t in tags),
                default=0.5
            )
            # Soft adjustment: multiply by (0.8 + 0.2 * min_conf)
            # → weak chain: ×0.8,  strong chain: ×1.0
            adj = 0.80 + 0.20 * min_conf
            if "ml_priority" in p:
                p["ml_priority"] = round(p["ml_priority"] * adj, 4)

        return roadmap

    # ── Print graph insights ──────────────────────────────────────────────────

    def print_graph_report(
        self,
        hidden_gaps: List[HiddenGap],
        final_conf: Dict[str, float],
        rating_predictor=None,
        user_elo: int = 1500,
    ):
        print("\n" + "=" * 62)
        print("🕸️  ML: TOPIC KNOWLEDGE GRAPH  (GNN Message Passing)")
        print("    Prerequisite chain analysis — finds HIDDEN mastery gaps")
        print("=" * 62)

        # ── Summary ───────────────────────────────────────────────────────────
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        print(f"\n   Graph: {n_nodes} topic nodes  |  {n_edges} prerequisite edges")

        # ── Strongest and weakest after propagation ────────────────────────────
        sorted_conf = sorted(final_conf.items(), key=lambda x: -x[1])
        top5  = sorted_conf[:5]
        bot5  = sorted_conf[-5:][::-1]

        print(f"\n   🏆 Strongest topics (GNN-adjusted confidence):")
        for tag, conf in top5:
            node = self.nodes.get(tag)
            raw  = node.raw_retention if node else conf
            delta = conf - raw
            delta_str = f" ({delta:+.0%} from prereq boost)" if abs(delta) > 0.05 else ""
            bar = self._bar(conf)
            print(f"      {bar}  {tag:<26}  conf={conf:.0%}{delta_str}")

        print(f"\n   ⚠️  Weakest topics (GNN-adjusted confidence):")
        for tag, conf in bot5:
            node = self.nodes.get(tag)
            raw  = node.raw_retention if node else conf
            delta = conf - raw
            delta_str = f" ({delta:+.0%} prereq drag)" if delta < -0.05 else ""
            bar = self._bar(conf)
            print(f"      {bar}  {tag:<26}  conf={conf:.0%}{delta_str}")

        # ── Hidden gaps ────────────────────────────────────────────────────────
        if hidden_gaps:
            print(f"\n   🔍 HIDDEN MASTERY GAPS detected:")
            for g in hidden_gaps[:4]:
                print(f"\n      Topic     : '{g.topic}'")
                print(f"      Apparent  : {g.apparent_retention:.0%} retention (per Ebbinghaus)")
                print(f"      True conf : {g.true_confidence:.0%} (after prerequisite propagation)")
                print(f"      Drag from : {', '.join(g.weak_prerequisites[:3])}")
                print(f"      💡 {g.insight}")
        else:
            print("\n   ✅ No hidden gaps detected — prerequisite chain is consistent.")

        # ── Prerequisite path examples ─────────────────────────────────────────
        print(f"\n   🗺️  Key prerequisite chains:")
        shown = set()
        for g in hidden_gaps[:2]:
            for wp in g.weak_prerequisites[:1]:
                chain = self._trace_path(wp, g.topic)
                if chain and tuple(chain) not in shown:
                    shown.add(tuple(chain))
                    arrow = " → ".join(chain)
                    print(f"      {arrow}")

        for src, dsts in list(self._adj_out.items())[:3]:
            if src in self.nodes:
                for dst, w in sorted(dsts, key=lambda x: -x[1])[:1]:
                    chain = (src, dst)
                    if chain not in shown:
                        shown.add(chain)
                        print(f"      '{src}' → '{dst}'  (weight={w:.2f})")

        # ── Rating trajectory ──────────────────────────────────────────────────
        if rating_predictor:
            rating_predictor.print_trajectory(user_elo, final_conf, self._profile)

    def _trace_path(self, src: str, dst: str, max_hops: int = 4) -> List[str]:
        """BFS to find shortest prereq path from src to dst."""
        from collections import deque
        if src not in self.nodes or dst not in self.nodes:
            return [src, dst]
        queue = deque([[src]])
        visited = {src}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == dst:
                return path
            if len(path) >= max_hops:
                continue
            for nxt, _ in self._adj_out.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return [src, dst]

    @staticmethod
    def _bar(v: float, w: int = 8) -> str:
        f = round(v * w)
        return "[" + "█" * f + "░" * (w - f) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# Rating Trajectory Predictor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RatingForecast:
    horizon_days: int
    predicted_elo: int
    elo_delta: int
    confidence_band: Tuple[int, int]   # (low, high)
    bottleneck_topic: Optional[str]
    milestone: str


class RatingTrajectoryPredictor:
    """
    Projects the user's competitive programming Elo rating forward in time
    using GNN-adjusted confidence scores + historical solve pace.

    Model:
      Effective Elo = f(mean_topic_confidence, solve_pace, contest_frequency)

      ΔElo/day = base_growth × (1 - current_normalized_skill) × pace_factor
        where:
          base_growth         = 3.0  (Elo points/day at median pace, 0% skill)
          current_norm_skill  = mean(final_conf) over contest-critical topics
          pace_factor         = recent_solves_per_day / 3.0 (normalised to 3/day)

      This is a saturating growth model: the closer you are to theoretical
      ceiling, the slower your rating grows per unit of practice.

    Uncertainty:
      σ(Δt) = base_σ × √Δt   (random walk variance component)
      confidence_band = (pred - 1.5σ, pred + 1.5σ)
    """

    BASE_GROWTH  = 3.0   # Elo/day at median pace
    BASE_SIGMA   = 15.0  # Elo volatility per day (scaled by √t)
    MAX_ELO      = 3500

    CONTEST_CRITICAL = {
        'dp', 'dynamic programming', 'greedy', 'graph', 'graphs',
        'binary search', 'two pointers', 'implementation', 'math',
        'constructive algorithms', 'trees', 'tree', 'dfs', 'bfs',
    }

    MILESTONES = [
        (800,  "Div 4 A–B problems"),
        (1000, "Div 4 B–C problems"),
        (1200, "Div 3 A–B problems"),
        (1400, "Div 3 C problems"),
        (1600, "Div 2 A–B problems"),
        (1800, "Div 2 C problems"),
        (2000, "Div 2 D problems"),
        (2200, "Div 2 D–E problems"),
        (2400, "Div 1 A problems"),
        (2600, "Div 1 B problems"),
    ]

    def predict(
        self,
        current_elo: int,
        final_conf: Dict[str, float],
        profile,
        horizons: List[int] = (7, 30, 90),
    ) -> List[RatingForecast]:

        # Current skill on contest-critical topics
        cc_confs = [v for k, v in final_conf.items() if k in self.CONTEST_CRITICAL]
        mean_cc  = float(np.mean(cc_confs)) if cc_confs else 0.50

        # Pace: recent solves per day (approximate from timeline event density)
        all_events = []
        for tl in profile.timelines.values():
            all_events.extend(e.timestamp for e in tl.events)

        if len(all_events) >= 4:
            all_events.sort()
            span_days = max((all_events[-1] - all_events[0]) / 86400, 1)
            pace = len(all_events) / span_days   # solves / day
        else:
            pace = 1.0

        pace_factor = float(np.clip(pace / 3.0, 0.2, 2.5))

        # Bottleneck: weakest contest-critical topic
        cc_weakest = min(
            ((k, v) for k, v in final_conf.items() if k in self.CONTEST_CRITICAL),
            key=lambda x: x[1],
            default=(None, 0.5),
        )
        bottleneck = cc_weakest[0]

        forecasts = []
        for days in horizons:
            # Saturating growth: Δelo = base × (1 - mean_cc) × pace × days × decay
            # decay prevents unrealistic long-term projections
            decay = math.exp(-days / 180)   # half-life of ~125 days
            elo_gain = (
                self.BASE_GROWTH
                * (1.0 - mean_cc)    # harder to improve as you get stronger
                * pace_factor
                * days
                * decay
            )
            pred_elo = int(min(current_elo + elo_gain, self.MAX_ELO))
            delta    = pred_elo - current_elo

            sigma  = self.BASE_SIGMA * math.sqrt(days)
            lo     = int(pred_elo - 1.5 * sigma)
            hi     = int(pred_elo + 1.5 * sigma)

            milestone = self._nearest_milestone(pred_elo, current_elo)

            forecasts.append(RatingForecast(
                horizon_days=days,
                predicted_elo=pred_elo,
                elo_delta=delta,
                confidence_band=(lo, hi),
                bottleneck_topic=bottleneck,
                milestone=milestone,
            ))

        return forecasts

    def _nearest_milestone(self, pred_elo: int, current_elo: int) -> str:
        # Find the next milestone above current that prediction is reaching
        next_ms = next(
            (label for elo, label in self.MILESTONES if elo > current_elo),
            None
        )
        if next_ms is None:
            return "World-class territory"
        ms_elo  = next((elo for elo, label in self.MILESTONES if label == next_ms), pred_elo)
        gap     = ms_elo - current_elo
        covered = pred_elo - current_elo
        if covered <= 0:
            return f"maintain current level ({next_ms} target)"
        pct = min(int(covered / max(gap, 1) * 100), 100)
        if pct >= 100:
            return f"✅ reach {next_ms}"
        return f"{pct}% toward {next_ms}"

    def print_trajectory(self, current_elo: int, final_conf: Dict[str, float], profile=None):
        # We need profile for pace; if not provided use defaults
        class _FakeProfile:
            timelines = {}
        if profile is None:
            profile = _FakeProfile()

        forecasts = self.predict(current_elo, final_conf, profile)

        print(f"\n   📈 RATING TRAJECTORY  (current Elo: ~{current_elo})")
        for f in forecasts:
            bar_filled = min(int((f.elo_delta / max(100, f.elo_delta + 1)) * 10), 10)
            bar = "▓" * bar_filled + "░" * (10 - bar_filled)
            print(f"      +{f.horizon_days:3d}d  [{bar}]  "
                  f"~{f.predicted_elo} Elo  "
                  f"(+{f.elo_delta:3d}, band {f.confidence_band[0]}–{f.confidence_band[1]})")
            print(f"              → {f.milestone}")
        if forecasts and forecasts[0].bottleneck_topic:
            print(f"\n      🔴 Bottleneck: '{forecasts[0].bottleneck_topic}' is "
                  f"limiting your rating growth most")
            print(f"         Fix this first for the fastest Elo gain.")