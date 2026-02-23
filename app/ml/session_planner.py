# ml/session_planner.py
"""
Neural Session Planner
======================
Four interconnected ML models that transform the roadmap from a static list
into an actionable, time-aware training plan.

Model 1 — RetentionForecaster
  Predicts on which future day each topic will cross a critical threshold
  by forward-integrating the fitted Ebbinghaus curve.  No new data needed —
  uses the curves already fitted by ForgettingCurveEstimator.

Model 2 — ThompsonSamplingBandit
  Treats each topic as an "arm" of a multi-armed bandit.  Each arm has a
  Beta(α, β) posterior over "solve-success probability".  α is initialised
  from historical success rate; β from failure rate.  Thompson Sampling
  draws a sample from each posterior and selects the topic with the highest
  expected learning gain × retention urgency.  This is the only component
  that actually updates online as the user solves problems in a session.

Model 3 — SpacedRepetitionScheduler  (SM-2 variant)
  Computes the ideal next review date for each topic using the SM-2 algorithm
  (the same algorithm behind Anki).  Uses difficulty estimate, current
  retention, and number of prior reviews to set the inter-repetition interval.
  Outputs: days_until_review, overdue_by_days, optimal_review_date.

Model 4 — ContestROIEstimator
  For each topic, estimates:
    expected_rating_delta(topic, Δt_practice_hours) =
        contest_frequency(topic) × solve_probability_gain(topic, Δt) × 50

  where solve_probability_gain is modelled as a saturating exponential
  (more practice has diminishing returns).  This lets us answer:
  "If I have 3 hours before the next contest, which topic is worth drilling?"

Orchestrator — SessionPlan
  Combines all four models into:
    • Forgetting curve forecast (7 / 14 / 30 day alert)
    • Avoidance detector (topic substitution analysis)
    • Optimal session problem set (knapsack on retention gain per minute)
    • 7-day contest prep calendar
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from ml.skill_trajectory import TemporalSkillProfile, TagTimeline, _ebbinghaus


# ─────────────────────────────────────────────────────────────────────────────
# Model 1  —  Retention Forecaster
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetentionForecast:
    tag: str
    current_retention: float
    days_to_critical: Optional[float]   # None = already critical / never decays below threshold
    days_to_warning: Optional[float]    # below 0.60
    predicted_retention_7d: float
    predicted_retention_14d: float
    predicted_retention_30d: float
    alert_level: str                     # "critical" | "warning" | "ok"


class RetentionForecaster:
    """
    Forward-integrates each fitted Ebbinghaus curve to find the future day
    on which retention will cross warning (0.60) and critical (0.40) thresholds.
    Uses binary search over the curve for efficiency.
    """
    WARNING_THRESHOLD  = 0.60
    CRITICAL_THRESHOLD = 0.40

    def forecast(self, tl: TagTimeline) -> RetentionForecast:
        lam = tl.decay_rate
        r   = 0.6   # default power — real fits stored on tl if available
        t0  = tl.last_seen_days   # days already elapsed since last practice

        def retention_at_future_day(extra_days: float) -> float:
            return float(_ebbinghaus(
                np.array([t0 + extra_days]), lam, r
            )[0])

        current = tl.retention
        pred_7d  = retention_at_future_day(7)
        pred_14d = retention_at_future_day(14)
        pred_30d = retention_at_future_day(30)

        def _days_to_threshold(threshold: float) -> Optional[float]:
            if current <= threshold:
                return 0.0   # already below
            # Binary search: find t such that R(t0 + t) = threshold
            lo, hi = 0.0, 3650.0   # search up to 10 years
            if retention_at_future_day(hi) > threshold:
                return None  # never decays that low in 10 years
            for _ in range(50):   # 50 iterations → precision < 0.0001 days
                mid = (lo + hi) / 2
                if retention_at_future_day(mid) > threshold:
                    lo = mid
                else:
                    hi = mid
            return round((lo + hi) / 2, 1)

        days_to_warning  = _days_to_threshold(self.WARNING_THRESHOLD)
        days_to_critical = _days_to_threshold(self.CRITICAL_THRESHOLD)

        if current < self.CRITICAL_THRESHOLD:
            alert = "critical"
        elif current < self.WARNING_THRESHOLD:
            alert = "warning"
        elif days_to_warning is not None and days_to_warning <= 7:
            alert = "warning"
        else:
            alert = "ok"

        return RetentionForecast(
            tag=tl.tag,
            current_retention=round(current, 3),
            days_to_critical=days_to_critical,
            days_to_warning=days_to_warning,
            predicted_retention_7d=round(pred_7d, 3),
            predicted_retention_14d=round(pred_14d, 3),
            predicted_retention_30d=round(pred_30d, 3),
            alert_level=alert,
        )

    def forecast_all(self, profile: TemporalSkillProfile) -> List[RetentionForecast]:
        forecasts = []
        for tag, tl in profile.timelines.items():
            forecasts.append(self.forecast(tl))
        return sorted(forecasts, key=lambda f: f.current_retention)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2  —  Thompson Sampling Bandit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BanditArm:
    tag: str
    alpha: float    # Beta prior: successes
    beta_: float    # Beta prior: failures
    n_attempts: int = 0
    n_successes: int = 0

    @property
    def mean_success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta_)

    def thompson_sample(self) -> float:
        """Draw one sample from Beta(α, β) posterior."""
        return float(beta_dist.rvs(self.alpha, self.beta_))

    def update(self, solved: bool):
        """Online Bayesian update after observing a solve outcome."""
        if solved:
            self.alpha    += 1
            self.n_successes += 1
        else:
            self.beta_ += 1
        self.n_attempts += 1


class ThompsonSamplingBandit:
    """
    Multi-armed bandit over topics.

    Each topic arm is initialised with:
      α = solved_recent + 1   (pseudo-count for successes)
      β = (total_in_catalog - solved_recent) / scaling + 1

    At selection time, we blend the Thompson sample with a
    "retention urgency bonus" so that topics currently decaying
    fast get extra selection pressure.
    """

    def __init__(self, seed: int = 42):
        self.arms: Dict[str, BanditArm] = {}
        random.seed(seed)
        np.random.seed(seed)

    def initialise_from_profile(
        self,
        profile: TemporalSkillProfile,
        tag_catalog_counts: Dict[str, int],   # how many catalog problems per tag
        forecasts: List[RetentionForecast],
    ):
        forecast_map = {f.tag: f for f in forecasts}

        for tag, tl in profile.timelines.items():
            n_solved  = len(tl.events)
            n_catalog = tag_catalog_counts.get(tag, max(n_solved * 3, 10))
            n_failed  = max(n_catalog - n_solved, 1)

            # Initialise Beta prior from observed history
            alpha = float(n_solved + 1)
            beta_ = float(n_failed / max(n_catalog / 20, 1) + 1)

            # Deflate alpha if retention is low (the user is forgetting it)
            fc = forecast_map.get(tag)
            if fc and fc.current_retention < 0.5:
                alpha *= fc.current_retention    # forgetting → lower effective success

            self.arms[tag] = BanditArm(
                tag=tag, alpha=alpha, beta_=beta_,
                n_attempts=n_solved, n_successes=n_solved
            )

        # Add any forecast-only tags not in timelines
        for fc in forecasts:
            if fc.tag not in self.arms:
                self.arms[fc.tag] = BanditArm(
                    tag=fc.tag, alpha=1.0, beta_=2.0)

    def select_topics(
        self,
        n: int,
        forecasts: List[RetentionForecast],
        weak_topics: List[str],
        contest_critical: set,
    ) -> List[Tuple[str, float]]:
        """
        Select the top-n topics for a practice session using Thompson Sampling
        blended with urgency scores.

        Returns list of (tag, composite_score) sorted descending.
        """
        if not self.arms:
            return []

        urgency_map = {f.tag: 1.0 - f.current_retention for f in forecasts}
        weak_set    = set(w.lower() for w in (weak_topics or []))

        scored = []
        for tag, arm in self.arms.items():
            ts_sample = arm.thompson_sample()

            urgency    = urgency_map.get(tag, 0.5)
            weak_bonus = 0.25 if tag in weak_set else 0.0
            cc_bonus   = 0.15 if tag in contest_critical else 0.0

            # Composite: TS sample weights expected performance gap;
            # urgency weights how badly retention is decaying
            composite = (0.50 * ts_sample +
                         0.35 * urgency   +
                         0.10 * weak_bonus +
                         0.05 * cc_bonus)

            scored.append((tag, round(composite, 4)))

        scored.sort(key=lambda x: -x[1])
        return scored[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Model 3  —  Spaced Repetition Scheduler  (SM-2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReviewSchedule:
    tag: str
    next_review_in_days: float
    overdue_by_days: float          # 0 if not overdue
    ease_factor: float              # SM-2 EF
    repetitions: int
    optimal_review_date: str        # human-readable "in X days" / "TODAY" / "X days overdue"


class SpacedRepetitionScheduler:
    """
    SM-2 algorithm adapted for competitive programming topics.

    Standard SM-2:
      I(1) = 1 day
      I(2) = 6 days
      I(n) = I(n-1) × EF,  EF ∈ [1.3, 2.5]
      EF' = EF + (0.1 - (5 - q) × (0.08 + (5 - q) × 0.02))
      where q ∈ {0,1,2,3,4,5} is quality of recall (0=blackout, 5=perfect)

    Adaptation:
      q is inferred from retention:
        R ≥ 0.90 → q=5,  R ≥ 0.75 → q=4,  R ≥ 0.55 → q=3,
        R ≥ 0.40 → q=2,  R ≥ 0.20 → q=1,  else → q=0
    """

    DEFAULT_EF = 2.5

    def _retention_to_quality(self, retention: float) -> int:
        if retention >= 0.90: return 5
        if retention >= 0.75: return 4
        if retention >= 0.55: return 3
        if retention >= 0.40: return 2
        if retention >= 0.20: return 1
        return 0

    def _ef_update(self, ef: float, q: int) -> float:
        new_ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        return max(1.3, min(2.5, new_ef))

    def schedule(self, tl: TagTimeline) -> ReviewSchedule:
        n_reps  = len(tl.events)
        current_r = tl.retention
        q = self._retention_to_quality(current_r)

        # Estimate EF from solve history — more consistent solves = higher EF
        if n_reps >= 4:
            # Use variance in inter-solve gaps as proxy for consistency
            ts = sorted([e.timestamp for e in tl.events])
            gaps = np.diff(ts) / 86400   # days
            gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-6))   # coefficient of variation
            ef = max(1.3, min(2.5, self.DEFAULT_EF - gap_cv * 0.3))
        else:
            ef = self.DEFAULT_EF

        # Compute SM-2 interval
        if n_reps == 0:
            interval = 1.0
        elif n_reps == 1:
            interval = 1.0
        elif n_reps == 2:
            interval = 6.0
        else:
            # Estimate prior interval from last gap
            if len(tl.events) >= 2:
                ts_sorted = sorted(e.timestamp for e in tl.events)
                last_gap  = (ts_sorted[-1] - ts_sorted[-2]) / 86400
                interval  = max(1.0, last_gap * ef)
            else:
                interval = 6.0 * (ef ** (n_reps - 2))

        # Adjust interval downward if quality is poor
        if q < 3:
            interval = 1.0   # relearn

        ef_new = self._ef_update(ef, q)

        # How many days until next review (from now)?
        next_review_in = max(0.0, interval - tl.last_seen_days)
        overdue = max(0.0, tl.last_seen_days - interval)

        if overdue > 1:
            date_str = f"{int(overdue)}d overdue — review TODAY"
        elif next_review_in < 1:
            date_str = "TODAY"
        elif next_review_in < 2:
            date_str = "tomorrow"
        else:
            date_str = f"in {int(next_review_in)}d"

        return ReviewSchedule(
            tag=tl.tag,
            next_review_in_days=round(next_review_in, 1),
            overdue_by_days=round(overdue, 1),
            ease_factor=round(ef_new, 2),
            repetitions=n_reps,
            optimal_review_date=date_str,
        )

    def schedule_all(self, profile: TemporalSkillProfile) -> List[ReviewSchedule]:
        schedules = []
        for tag, tl in profile.timelines.items():
            schedules.append(self.schedule(tl))
        return sorted(schedules, key=lambda s: s.next_review_in_days)


# ─────────────────────────────────────────────────────────────────────────────
# Model 4  —  Contest ROI Estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContestROI:
    tag: str
    roi_per_hour: float              # expected rating delta per practice hour
    solve_prob_now: float            # probability of solving a contest problem with this tag
    solve_prob_after_1h: float       # after 1h of targeted practice
    solve_prob_after_3h: float       # after 3h of targeted practice
    recommended_hours: float         # optimal hours to invest before next contest
    priority_rank: int               # 1 = highest ROI


# Contest topic frequency (fraction of CF Div2/3 contests with this tag in ≥1 problem)
CONTEST_FREQ = {
    'implementation': 0.95, 'math': 0.90, 'greedy': 0.85,
    'constructive algorithms': 0.80, 'brute force': 0.75,
    'dp': 0.70, 'binary search': 0.65, 'sorting': 0.60,
    'sortings': 0.60, 'two pointers': 0.55, 'string': 0.50,
    'strings': 0.50, 'graphs': 0.45, 'graph': 0.45,
    'trees': 0.40, 'tree': 0.40, 'dfs': 0.35, 'bfs': 0.30,
    'number theory': 0.30, 'data structures': 0.35, 'dsu': 0.25,
    'segment tree': 0.20, 'bit manipulation': 0.25, 'combinatorics': 0.30,
}


class ContestROIEstimator:
    """
    Models expected contest rating gain from investing time in each topic.

    Solve probability model:
      p_solve(tag, t) = p0 + (p_max - p0) × (1 - exp(-k × t))
      where:
        p0    = current solve probability (from retention + solve history)
        p_max = ceiling (0.95 for fundamentals, 0.80 for advanced)
        k     = learning rate (from ForgettingCurveEstimator.learning_rate_est)
        t     = practice hours

    ROI model:
      ROI(tag, t) = contest_freq(tag) × (p_solve(t) - p0) × 50  (50 ≈ avg rating per problem)
    """

    RATING_PER_SOLVE = 50.0   # approximate Elo delta per solved contest problem

    FUNDAMENTAL_TAGS = {
        'implementation', 'math', 'greedy', 'constructive algorithms',
        'brute force', 'sorting', 'sortings', 'string', 'strings',
        'two pointers', 'binary search',
    }

    def estimate(self, tl: TagTimeline, weak_topics: List[str]) -> ContestROI:
        tag   = tl.tag
        freq  = CONTEST_FREQ.get(tag, 0.15)
        p_max = 0.92 if tag in self.FUNDAMENTAL_TAGS else 0.78
        lr    = max(tl.learning_rate_est, 0.1)   # solves/day → use as proxy for learning speed

        # p0: current solve probability
        # Blend retention (memory component) with success rate (skill component)
        success_rate = len(tl.events) / max(len(tl.events) + 2, 1)
        p0 = float(np.clip(0.6 * tl.retention + 0.4 * success_rate, 0.05, 0.95))

        # Learning rate constant: how many hours to reach 63% of max gain
        # Learning rate ×3 means 3× faster pace → k is proportional
        k = 0.8 * lr   # steeper for faster learners

        def p_at_t(hours: float) -> float:
            return p0 + (p_max - p0) * (1 - math.exp(-k * hours))

        p1h = p_at_t(1.0)
        p3h = p_at_t(3.0)

        # Optimal hours: point of diminishing returns (≥90% of max gain)
        # Solve: (p_max - p0) × (1 - exp(-k × t_opt)) = 0.9 × (p_max - p0)
        # → t_opt = -ln(0.1) / k
        t_opt = -math.log(0.1) / max(k, 0.01)
        t_opt = round(min(t_opt, 6.0), 1)   # cap at 6 hours

        # ROI per hour (marginal gain at 1 hour)
        roi = freq * (p1h - p0) * self.RATING_PER_SOLVE

        return ContestROI(
            tag=tag,
            roi_per_hour=round(roi, 3),
            solve_prob_now=round(p0, 3),
            solve_prob_after_1h=round(p1h, 3),
            solve_prob_after_3h=round(p3h, 3),
            recommended_hours=t_opt,
            priority_rank=0,  # set after sorting
        )

    def rank_all(
        self, profile: TemporalSkillProfile, weak_topics: List[str]
    ) -> List[ContestROI]:
        results = []
        for tag, tl in profile.timelines.items():
            results.append(self.estimate(tl, weak_topics))
        results.sort(key=lambda r: -r.roi_per_hour)
        for i, r in enumerate(results):
            r.priority_rank = i + 1
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Avoidance Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AvoidanceSignal:
    avoided_topic: str
    substitute_topic: str
    avoidance_score: float   # 0–1
    evidence: str


class AvoidanceDetector:
    """
    Detects topic substitution: the user consistently picks problems tagged
    with substitute_topic instead of avoided_topic, even when both are in
    their weak-topic list.

    Method: for each pair of weak topics, compute the ratio of recent solves.
    A ratio >> 1 suggests the user avoids the topic with fewer solves.
    We require the pair to be semantically similar (co-occur in the embedding)
    to filter out false positives (e.g. "dp vs graph" aren't substitutes).
    """

    def detect(
        self,
        profile: TemporalSkillProfile,
        weak_topics: List[str],
        embedder,          # TopicEmbedder
    ) -> List[AvoidanceSignal]:
        signals = []
        weak_set = [t.lower() for t in weak_topics]

        for i, tag_a in enumerate(weak_set):
            for tag_b in weak_set[i + 1:]:
                tl_a = profile.timelines.get(tag_a)
                tl_b = profile.timelines.get(tag_b)

                n_a = len(tl_a.events) if tl_a else 0
                n_b = len(tl_b.events) if tl_b else 0

                if n_a + n_b < 3:
                    continue

                ratio = (n_a + 1) / (n_b + 1)
                if ratio < 3 and ratio > 0.33:
                    continue   # roughly equal — no strong signal

                avoided    = tag_b if ratio > 3 else tag_a
                substitute = tag_a if ratio > 3 else tag_b
                avoidance_score = min(abs(math.log(ratio)) / math.log(10), 1.0)

                # Check semantic proximity (only flag if they're in adjacent territory)
                sim_score = 0.0
                emb_a = embedder.get_embedding(tag_a)
                emb_b = embedder.get_embedding(tag_b)
                if emb_a is not None and emb_b is not None:
                    sim_score = float(emb_a @ emb_b)

                if sim_score < 0.10:
                    continue   # not adjacent enough to be substitutes

                n_sub = len(profile.timelines[substitute].events) if substitute in profile.timelines else 0
                n_avd = len(profile.timelines[avoided].events) if avoided in profile.timelines else 0
                evidence = (
                    f"{n_sub} recent '{substitute}' solves vs "
                    f"{n_avd} '{avoided}' solves (sim={sim_score:.2f})"
                )

                signals.append(AvoidanceSignal(
                    avoided_topic=avoided,
                    substitute_topic=substitute,
                    avoidance_score=round(avoidance_score, 3),
                    evidence=evidence,
                ))

        signals.sort(key=lambda s: -s.avoidance_score)
        return signals[:4]


# ─────────────────────────────────────────────────────────────────────────────
# Session Planner  —  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DayPlan:
    day: int               # 1–7
    label: str             # "Monday" etc.
    focus_topics: List[str]
    target_problems: List[dict]
    session_goal: str
    expected_retention_gain: float


class SessionPlanner:
    """
    Orchestrates all four models to produce:
      1. Forgetting curve alert report (7/14/30 day forecast)
      2. Topic avoidance analysis
      3. Optimal N-hour session plan (knapsack on expected retention gain)
      4. 7-day contest prep calendar
    """

    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def __init__(
        self,
        profile: TemporalSkillProfile,
        embedder,
        roadmap: List[dict],
        weak_topics: List[str],
        contest_penalty: float,
    ):
        self.profile  = profile
        self.embedder = embedder
        self.roadmap  = roadmap
        self.weak_topics = weak_topics
        self.contest_penalty = contest_penalty

        self.forecaster  = RetentionForecaster()
        self.bandit      = ThompsonSamplingBandit()
        self.sm2         = SpacedRepetitionScheduler()
        self.roi_model   = ContestROIEstimator()
        self.avoidance   = AvoidanceDetector()

        # Pre-compute
        self.forecasts   = self.forecaster.forecast_all(profile)
        self.schedules   = self.sm2.schedule_all(profile)
        self.roi_ranks   = self.roi_model.rank_all(profile, weak_topics)

        # Build tag → catalog count for bandit init
        tag_counts: Dict[str, int] = defaultdict(int)
        for p in roadmap:
            for t in p.get("tags", []):
                tag_counts[t.lower()] += 1

        contest_critical = {
            'dp', 'dynamic programming', 'graph', 'graphs', 'tree', 'trees',
            'two pointers', 'binary search', 'dfs', 'bfs', 'greedy',
            'backtracking', 'sliding window', 'heap', 'priority queue'
        }
        self.bandit.initialise_from_profile(profile, tag_counts, self.forecasts)
        self.bandit_picks = self.bandit.select_topics(
            n=8, forecasts=self.forecasts,
            weak_topics=weak_topics, contest_critical=contest_critical
        )

        self.avoidance_signals = self.avoidance.detect(profile, weak_topics, embedder)

    # ── 1. Forecast report ───────────────────────────────────────────────────

    def print_forecast_report(self):
        print("\n" + "=" * 62)
        print("🔮 ML: RETENTION FORECAST — UPCOMING ALERTS")
        print("    (Forward-integrating your personalised Ebbinghaus curves)")
        print("=" * 62)

        critical = [f for f in self.forecasts if f.alert_level == "critical"]
        warning  = [f for f in self.forecasts if f.alert_level == "warning"]
        ok       = [f for f in self.forecasts if f.alert_level == "ok"]

        if critical:
            print(f"\n🚨 CRITICAL — already below 40% retention:")
            for f in critical[:5]:
                d_c = f"→ critical in {f.days_to_critical:.0f}d" if f.days_to_critical else ""
                bar = self._retention_bar(f.current_retention)
                print(f"   {bar}  {f.tag:<26}  "
                      f"now={f.current_retention:.0%}  "
                      f"7d={f.predicted_retention_7d:.0%}  "
                      f"14d={f.predicted_retention_14d:.0%}")

        if warning:
            print(f"\n⚠️  WARNING — will fall critical within 7–14 days:")
            for f in warning[:5]:
                d_c = f.days_to_critical
                bar = self._retention_bar(f.current_retention)
                crit_str = f"  critical in {d_c:.0f}d" if d_c else ""
                print(f"   {bar}  {f.tag:<26}  "
                      f"now={f.current_retention:.0%}  "
                      f"7d={f.predicted_retention_7d:.0%}  "
                      f"30d={f.predicted_retention_30d:.0%}"
                      f"{crit_str}")

        decaying = [f for f in self.forecasts
                    if f.alert_level == "ok"
                    and f.predicted_retention_30d < 0.60
                    and f.current_retention >= 0.60]
        if decaying:
            print(f"\n📉 WATCH — will reach warning zone within 30 days:")
            for f in sorted(decaying, key=lambda x: x.predicted_retention_30d)[:4]:
                bar = self._retention_bar(f.current_retention)
                print(f"   {bar}  {f.tag:<26}  "
                      f"now={f.current_retention:.0%}  "
                      f"→30d={f.predicted_retention_30d:.0%}")

        total_at_risk = len(critical) + len(warning)
        if total_at_risk == 0:
            print("\n   ✅ All tracked topics are within healthy retention range.")

    @staticmethod
    def _retention_bar(r: float, w: int = 8) -> str:
        f = round(r * w)
        return "[" + "█" * f + "░" * (w - f) + "]"

    # ── 2. Avoidance report ──────────────────────────────────────────────────

    def print_avoidance_report(self):
        if not self.avoidance_signals:
            return
        print("\n" + "=" * 62)
        print("🧠 ML: TOPIC AVOIDANCE DETECTOR")
        print("    (Statistically detects substitution patterns in your history)")
        print("=" * 62)
        for s in self.avoidance_signals:
            bar_score = "▓" * int(s.avoidance_score * 10) + "░" * (10 - int(s.avoidance_score * 10))
            print(f"\n   [{bar_score}]  avoidance score = {s.avoidance_score:.2f}")
            print(f"   You tend to solve '{s.substitute_topic}' instead of '{s.avoided_topic}'")
            print(f"   Evidence: {s.evidence}")
            print(f"   💡 Try: force yourself to attempt '{s.avoided_topic}' problems next session")

    # ── 3. Session plan ──────────────────────────────────────────────────────

    def build_session_plan(self, hours: float = 2.0) -> List[dict]:
        """
        Knapsack: given `hours`, select problems from the roadmap that
        maximise total expected retention gain.

        Time budget per problem:
          easy (800)   → 20 min
          medium (1200)→ 35 min
          hard (1500+) → 55 min

        Expected retention gain per problem:
          = urgency(tag) × difficulty_stretch × topic_bandit_score
        """
        budget_min = hours * 60
        time_per_diff = {800: 20, 1200: 35, 1400: 45, 1500: 50, 1600: 55, 1800: 65, 2000: 80}

        bandit_score_map = dict(self.bandit_picks)
        sm2_map = {s.tag: s for s in self.schedules}

        candidates = []
        for p in self.roadmap:
            tags = [t.lower() for t in p.get("tags", [])]
            diff = int(p.get("difficulty", 1200))
            t_est = min((t for d, t in time_per_diff.items() if d >= diff),
                        default=80)

            urgency   = max((1 - self.profile.timelines[t].retention
                             if t in self.profile.timelines else 0.5)
                            for t in tags) if tags else 0.5

            bandit_s  = max((bandit_score_map.get(t, 0.3) for t in tags), default=0.3)
            stretch   = 1.0 + (diff - 800) / 1600   # harder = more gain

            # SM-2 overdue bonus: if the topic is overdue for review, add urgency
            overdue_bonus = max(
                (min(sm2_map[t].overdue_by_days / 7, 0.5) if t in sm2_map else 0.0)
                for t in tags
            )

            gain = urgency * bandit_s * stretch + overdue_bonus

            candidates.append({
                **p,
                "est_minutes": t_est,
                "session_gain": round(gain, 4),
                "session_reason": self._session_reason(tags, diff, urgency, bandit_s)
            })

        # Greedy knapsack (sort by gain/time ratio)
        candidates.sort(key=lambda x: -x["session_gain"] / x["est_minutes"])
        selected, total_time = [], 0.0
        for c in candidates:
            if total_time + c["est_minutes"] <= budget_min:
                selected.append(c)
                total_time += c["est_minutes"]
            if total_time >= budget_min * 0.90:
                break

        return selected

    def _session_reason(self, tags: List[str], diff: int, urgency: float, bandit_s: float) -> str:
        sm2_map = {s.tag: s for s in self.schedules}
        overdue_tags = [t for t in tags if t in sm2_map and sm2_map[t].overdue_by_days > 1]
        if overdue_tags:
            return f"SM-2 review overdue for '{overdue_tags[0]}'"
        if urgency > 0.7:
            urgent_tag = next((t for t in tags if t in self.profile.timelines
                               and (1 - self.profile.timelines[t].retention) > 0.7), tags[0] if tags else "?")
            return f"'{urgent_tag}' retention critically low — reinforce now"
        if bandit_s > 0.6:
            return f"Thompson sampling: '{tags[0]}' has highest expected learning gain"
        return f"balanced difficulty progression at {diff}"

    def print_session_plan(self, hours: float = 2.0):
        session = self.build_session_plan(hours)
        total_min = sum(p["est_minutes"] for p in session)
        print("\n" + "=" * 62)
        print(f"⚡ ML: OPTIMAL {hours:.0f}-HOUR SESSION PLAN")
        print(f"    (Knapsack optimised via Thompson Sampling + SM-2 urgency)")
        print(f"    Estimated time: {total_min} min | {len(session)} problems")
        print("=" * 62)

        cum_time = 0
        for i, p in enumerate(session, 1):
            cum_time += p["est_minutes"]
            src  = "🔷" if p["source"] == "LeetCode" else "🔶"
            weak = " ⚠️" if p.get("matches_weak_topic") else ""
            tags = ", ".join(p.get("tags", [])[:3])
            print(f"\n   {i}. {src} [{p['source']}] {p['name']}{weak}")
            print(f"      Difficulty : {p['difficulty']}  |  ~{p['est_minutes']} min  "
                  f"(cumulative: {cum_time} min)")
            print(f"      Tags       : {tags}")
            print(f"      📌 Why now : {p['session_reason']}")
            if p.get("link"):
                print(f"      Link       : {p['link']}")

        print(f"\n   💡 Complete this session to gain an estimated "
              f"+{sum(p['session_gain'] for p in session):.2f} retention units")

    # ── 4. 7-day contest prep calendar ───────────────────────────────────────

    def build_contest_calendar(self) -> List[DayPlan]:
        """
        Build a 7-day calendar using ContestROI rankings + SM-2 schedule.
        Day 1–2: critical retention recovery (overdue reviews)
        Day 3–5: highest-ROI topics (contest impact)
        Day 6–7: simulation practice (mixed topics at contest difficulty)
        """
        roi_map     = {r.tag: r for r in self.roi_ranks}
        sm2_map     = {s.tag: s for s in self.schedules}
        overdue_tags = sorted(
            [s for s in self.schedules if s.overdue_by_days > 0],
            key=lambda s: -s.overdue_by_days
        )
        top_roi_tags = [r.tag for r in self.roi_ranks[:6] if r.roi_per_hour > 0.5]

        calendar = []
        import datetime
        today = datetime.date.today()

        for day_idx in range(7):
            day_label = self.DAYS[(today.weekday() + day_idx) % 7]
            date_str  = (today + datetime.timedelta(days=day_idx)).strftime("%b %d")

            if day_idx < 2 and overdue_tags:
                # Days 1–2: overdue SM-2 reviews
                focus = [t.tag for t in overdue_tags[: min(3, len(overdue_tags))]]
                goal  = "Spaced repetition review — recover fading topics"
                problems = self._pick_problems_for_topics(focus, difficulty_max=1400, n=4)
                roi_gain = sum(roi_map[t].roi_per_hour for t in focus if t in roi_map)

            elif day_idx in (2, 3, 4) and top_roi_tags:
                # Days 3–5: highest contest ROI
                idx       = day_idx - 2
                day_focus = top_roi_tags[idx * 2: idx * 2 + 2]
                focus     = day_focus if day_focus else top_roi_tags[:2]
                goal      = "Contest ROI maximisation — highest expected rating gain"
                diff_target = 1400 + idx * 200   # escalate: 1400→1600→1800
                problems = self._pick_problems_for_topics(focus, difficulty_max=diff_target, n=4)
                roi_gain = sum(roi_map[t].roi_per_hour for t in focus if t in roi_map)

            else:
                # Days 6–7: mixed contest simulation
                sim_focus = [r.tag for r in self.roi_ranks[:4]]
                focus     = sim_focus
                goal      = "Contest simulation — mixed topics at speed"
                problems  = self._pick_problems_for_topics(focus, difficulty_max=2000, n=5)
                roi_gain  = sum(roi_map[t].roi_per_hour for t in focus if t in roi_map)

            calendar.append(DayPlan(
                day=day_idx + 1,
                label=f"{day_label} ({date_str})",
                focus_topics=focus,
                target_problems=problems,
                session_goal=goal,
                expected_retention_gain=round(roi_gain, 2),
            ))

        return calendar

    def _pick_problems_for_topics(
        self, topics: List[str], difficulty_max: int, n: int
    ) -> List[dict]:
        topic_set = set(t.lower() for t in topics)
        matches = [
            p for p in self.roadmap
            if any(t.lower() in topic_set for t in p.get("tags", []))
            and int(p.get("difficulty", 1200)) <= difficulty_max
        ]
        matches.sort(key=lambda p: -p.get("ml_priority", p.get("session_gain", 0.5)))
        return matches[:n]

    def print_contest_calendar(self):
        calendar = self.build_contest_calendar()
        print("\n" + "=" * 62)
        print("📅 ML: 7-DAY CONTEST PREP CALENDAR")
        print("    (Contest ROI model + SM-2 schedule + Thompson Sampling)")
        print("=" * 62)

        for plan in calendar:
            print(f"\n  ━━ Day {plan.day}: {plan.label} ━━")
            print(f"  🎯 Goal     : {plan.session_goal}")
            print(f"  📌 Focus    : {', '.join(plan.focus_topics)}")
            if plan.expected_retention_gain > 0:
                print(f"  📈 Est. ROI : +{plan.expected_retention_gain:.1f} rating points/hr")
            if plan.target_problems:
                print(f"  📋 Problems :")
                for p in plan.target_problems[:3]:
                    src  = "🔷" if p["source"] == "LeetCode" else "🔶"
                    diff = p["difficulty"]
                    name = p["name"][:38]
                    print(f"       {src} {name:<38}  d={diff}")
            else:
                print(f"  📋 (no targeted problems available — free practice day)")

    # ── 5. SM-2 review schedule ──────────────────────────────────────────────

    def print_review_schedule(self):
        due_soon = [s for s in self.schedules
                    if s.next_review_in_days <= 3 or s.overdue_by_days > 0]
        if not due_soon:
            return
        print("\n" + "=" * 62)
        print("🗓️  ML: SPACED REPETITION REVIEW SCHEDULE  (SM-2)")
        print("    (Your personalised Anki-style review dates)")
        print("=" * 62)
        print()
        for s in sorted(due_soon, key=lambda x: (x.overdue_by_days * -1, x.next_review_in_days))[:8]:
            overdue_str = f" ⚠️ {s.overdue_by_days:.0f}d overdue" if s.overdue_by_days > 0 else ""
            print(f"   • {s.tag:<28}  review: {s.optimal_review_date:<22}  "
                  f"EF={s.ease_factor:.2f}  reps={s.repetitions}{overdue_str}")

    # ── Print all ────────────────────────────────────────────────────────────

    def print_all(self, session_hours: float = 2.0):
        self.print_forecast_report()
        self.print_review_schedule()
        self.print_avoidance_report()
        self.print_session_plan(hours=session_hours)
        self.print_contest_calendar()