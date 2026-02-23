# main.py  (ML-enhanced version)
"""
Drop-in replacement for the original main.py.
Adds the ML pipeline after the standard roadmap generation.
"""

import pandas as pd
from collections import Counter
import sys
import time

from app.api.leetcode_api import (
    fetch_leetcode_submissions,
    fetch_leetcode_solved_stats,
    fetch_leetcode_contests
)
from app.api.codeforces_api import (
    fetch_cf_submissions,
    fetch_cf_contests
)

from preprocess.leetcode_catalog import load_leetcode_catalog
from preprocess.codeforces_catalog import load_cf_catalog
from preprocess.normalize import normalize_cf, normalize_lc

from analysis.weakness_analysis import detect_weak_topics
from analysis.contest_analysis import (
    contest_penalty_cf,
    contest_penalty_lc
)
from recommender.roadmap import generate_roadmap

# ── NEW: ML imports ─────────────────────────────────────────────────────────
try:
    from ml.pipeline import MLPipeline
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ML features unavailable: {e}")
    ML_AVAILABLE = False
# ────────────────────────────────────────────────────────────────────────────

# -------------------------
# USER INPUT
# -------------------------
lc_user = input("Enter LeetCode username: ").strip()
cf_user = input("Codeforces handle: ").strip()

if not lc_user and not cf_user:
    print("❌ Error: You must provide at least one username!")
    sys.exit(1)

# -------------------------
# FETCH USER DATA
# -------------------------
print("\n🔄 Fetching user data...")

lc_subs = []
lc_contests = []
cf_subs = []
cf_contests = []
lc_total_solved = 0
cf_total_solved = 0
solved_lc_recent = set()
solved_cf = set()

time.sleep(1)

if lc_user:
    print("   • Fetching LeetCode data...")
    lc_subs = fetch_leetcode_submissions(lc_user, limit=500)
    lc_total_solved, lc_stats = fetch_leetcode_solved_stats(lc_user)

    solved_lc_recent = set(s.get("titleSlug") for s in lc_subs
                           if s.get("statusDisplay") == "Accepted")

    easy = lc_stats[0] if lc_stats and len(lc_stats) > 0 else 0
    medium = lc_stats[1] if lc_stats and len(lc_stats) > 1 else 0
    hard = lc_stats[2] if lc_stats and len(lc_stats) > 2 else 0
    print(f"   📊 LC stats: E={easy}, M={medium}, H={hard}")

    lc_contests = fetch_leetcode_contests(lc_user)
    time.sleep(1)

if cf_user:
    print("   • Fetching Codeforces data...")
    cf_subs = fetch_cf_submissions(cf_user)
    cf_total_solved = sum(1 for s in cf_subs if s.get("verdict") == "OK")

    for s in cf_subs:
        if s.get("verdict") == "OK":
            p = s.get("problem", {})
            if p.get("contestId") and p.get("index"):
                solved_cf.add(f"{p['contestId']}{p['index']}")

    cf_contests = fetch_cf_contests(cf_user)

# -------------------------
# LOAD CATALOGS
# -------------------------
print("\n📚 Loading problem catalogs...")
lc_df = normalize_lc(load_leetcode_catalog()) if lc_user else pd.DataFrame()
cf_df = normalize_cf(load_cf_catalog()) if cf_user else pd.DataFrame()
print(f"   ✓ LC catalog: {len(lc_df)} problems")
print(f"   ✓ CF catalog: {len(cf_df)} problems")

# -------------------------
# DEBUG INFO
# -------------------------
print(f"📋 Debug: LC recent={len(solved_lc_recent)}, total={lc_total_solved} | "
      f"CF recent={len(solved_cf)}, total={cf_total_solved}")

# -------------------------
# USER LEVEL
# -------------------------
total_solved = lc_total_solved + cf_total_solved

if total_solved == 0:
    user_level = "Absolute Beginner"
elif total_solved < 25:
    user_level = "Beginner"
elif total_solved < 100:
    user_level = "Intermediate"
elif total_solved < 300:
    user_level = "Advanced"
else:
    user_level = "Expert"

# -------------------------
# WEAK TOPICS
# -------------------------
print("\n🔍 Analyzing weak topics...")
weak_topics = detect_weak_topics(
    cf_df=cf_df,
    lc_df=lc_df,
    solved_cf=solved_cf,
    solved_lc=solved_lc_recent,
    cf_contests=cf_contests,
    lc_contests=lc_contests,
    lc_total_solved=lc_total_solved,
    threshold=0.35,
    min_attempts=8,
    adaptive_min=True
)

# -------------------------
# CONTEST PERFORMANCE
# -------------------------
print("🏆 Analyzing contest performance...")
cf_penalty = contest_penalty_cf(cf_contests) if cf_contests else 0.0
lc_penalty = contest_penalty_lc(lc_contests) if lc_contests else 0.0
combined_penalty = round((cf_penalty + lc_penalty) / 2, 3) if cf_contests or lc_contests else 0.5

# -------------------------
# OUTPUT ANALYSIS
# -------------------------
print("\n" + "=" * 70)
print("📊 PERFORMANCE ANALYSIS")
print("=" * 70)
print(f"\n👤 User Level: {user_level}")

print(f"\n✅ Problems Solved:")
print(f"   • LeetCode: {lc_total_solved} total ({len(solved_lc_recent)} recent)")
print(f"   • Codeforces: {cf_total_solved} total ({len(solved_cf)} recent)")
print(f"   • Combined Total: {total_solved}")

active_platforms = []
if lc_total_solved > 0:
    active_platforms.append("LeetCode")
if cf_total_solved > 0:
    active_platforms.append("Codeforces")

if len(active_platforms) == 1:
    print(f"\n💡 Active Platform: {active_platforms[0]} only")
    inactive = "Codeforces" if active_platforms[0] == "LeetCode" else "LeetCode"
    print(f"   Consider trying {inactive} for variety!")

print(f"\n🎯 Weak Topics (need improvement):")
if weak_topics:
    fundamental = ['array', 'string', 'math', 'hash table', 'two pointers']
    weak_fund = [t for t in weak_topics if t in fundamental]
    weak_adv = [t for t in weak_topics if t not in fundamental]

    if weak_fund:
        print("   📌 Fundamental Topics:")
        for i, topic in enumerate(weak_fund[:5], 1):
            print(f"      {i}. {topic}")
    if weak_adv:
        print("   🔥 Advanced Topics:")
        for i, topic in enumerate(weak_adv[:5], 1):
            print(f"      {i}. {topic}")
else:
    print("   🎉 Great job! No major weaknesses detected.")

print(f"\n📉 Contest Performance:")
if cf_contests or lc_contests:
    if cf_contests:
        print(f"   • Codeforces: {len(cf_contests)} contests (penalty: {round(cf_penalty, 3)})")
    if lc_contests:
        print(f"   • LeetCode: {len(lc_contests)} contests (penalty: {round(lc_penalty, 3)})")
    print(f"   • Combined penalty: {combined_penalty}")
else:
    print("   No contest history - consider joining contests!")

# -------------------------
# STANDARD ROADMAP
# -------------------------
print("\n🗺️  Generating personalized roadmap...")
roadmap = generate_roadmap(
    lc_df=lc_df,
    cf_df=cf_df,
    weak_topics=weak_topics,
    solved_lc=solved_lc_recent,
    solved_cf=solved_cf,
    contest_penalty=combined_penalty,
    max_items=50,
    balance_platforms=(cf_total_solved > 0)
)

if not roadmap:
    print("\n❌ Could not generate roadmap. Please check your internet connection.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# ML ENHANCEMENT BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if ML_AVAILABLE:
    try:
        ml = MLPipeline()
        roadmap = ml.run(
            lc_subs=lc_subs,
            cf_subs=cf_subs,
            lc_df=lc_df if lc_user else None,
            cf_df=cf_df if cf_user else None,
            roadmap=roadmap,
            weak_topics=weak_topics,
            contest_penalty=combined_penalty,
            lc_total_solved=lc_total_solved,
            cf_total_solved=cf_total_solved,
        )

        # Estimate user Elo from solve count + contest penalty
        _elo_base = min(800 + total_solved // 3, 2200)
        _elo = max(800, int(_elo_base * (1.0 - combined_penalty * 0.3)))

        try:
            _hours_input = input("\n⏱️  How many hours do you have for practice today? [default: 2]: ").strip()
            _session_hours = float(_hours_input) if _hours_input else 2.0
        except (ValueError, EOFError):
            _session_hours = 2.0

        ml.print_insights(user_elo=_elo, session_hours=_session_hours)
    except Exception as e:
        import traceback
        print(f"\n⚠️  ML pipeline error (falling back to standard roadmap): {e}")
        traceback.print_exc()
# ─────────────────────────────────────────────────────────────────────────────

# -------------------------
# FINAL OUTPUT
# -------------------------
pd.set_option("display.max_colwidth", None)
PRINT_LIMIT = 40

platform_count = Counter(p['source'] for p in roadmap[:PRINT_LIMIT])
print("\n" + "=" * 60)
print(f"📌 ML-ENHANCED PERSONALIZED ROADMAP (Top {PRINT_LIMIT})")
print("=" * 60)
print(f"Platform Distribution: {dict(platform_count)}")

# Show whether ML reranking was applied
if ML_AVAILABLE and any("ml_priority" in p for p in roadmap[:3]):
    print("🤖 Ordering optimised by Neural Roadmap Reranker")
print("=" * 60 + "\n")

for i, row in enumerate(roadmap[:PRINT_LIMIT], start=1):
    emoji = "🔷" if row['source'] == "LeetCode" else "🔶"
    tags_s = ", ".join(row.get("tags", [])[:5]) or "n/a"
    weak_indicator = " ⚠️ WEAK TOPIC" if row.get("matches_weak_topic") else ""

    print(f"{i}. {emoji} [{row['source']}] {row['name']}{weak_indicator}")
    print(f"   Difficulty : {row['difficulty']}")
    print(f"   Tags       : {tags_s}")

    # ── NEW: ML insight line ─────────────────────────────────────────────
    if row.get("ml_explanation"):
        print(f"   🧠 ML Reason: {row['ml_explanation']}")
        forgetting = row.get("forgetting_urgency", 0)
        if forgetting > 0.6:
            print(f"   ⏳ Retention risk: {forgetting:.0%} forgotten — review now!")
    # ────────────────────────────────────────────────────────────────────

    if row.get("link"):
        print(f"   Link       : {row['link']}")
    print()

print("=" * 60)
if weak_topics:
    print("💡 Focus on problems marked with ⚠️ WEAK TOPIC first!")
    print("🚀 These target your actual performance gaps!")
else:
    print("💡 Work through these problems to maintain your Expert level!")
if ML_AVAILABLE:
    print("🤖 Ordering powered by forgetting curves + topic embeddings + SA")
print("=" * 60)