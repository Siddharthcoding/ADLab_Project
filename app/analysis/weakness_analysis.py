# analysis/weakness_analysis.py

from collections import defaultdict, Counter
from analysis.contest_analysis import contest_penalty_cf, contest_penalty_lc

def detect_weak_topics(
    cf_df,
    lc_df,
    solved_cf,
    solved_lc,
    cf_contests=None,
    lc_contests=None,
    lc_total_solved=0, 
    threshold=0.3,
    min_attempts=5,
    adaptive_min=True
):
    """
    FIXED: Expert-aware weak topic detection. Uses recent solves for matching,
    total solves for calibration. Prioritizes CONTEST-critical topics for experts.
    """
    
    tag_stats = defaultdict(lambda: {
        'total': 0, 
        'solved_recent': 0,
        'attempted_recent': 0,
        'cf_problems': [],
        'lc_problems': [],
        'contest_weight': 1.0
    })

    # ---------- PROCESS CATALOGS (unchanged) ----------
    if cf_df is not None and not cf_df.empty:
        for _, row in cf_df.iterrows():
            pid = row.get("problem_id")
            tags = row.get("tags", [])
            tags = [str(t).strip().lower() for t in tags if t and str(t).strip()]
            
            for tag in tags:
                tag_stats[tag]['total'] += 1
                tag_stats[tag]['cf_problems'].append(pid)
                if pid and pid in solved_cf:
                    tag_stats[tag]['solved_recent'] += 1
                    tag_stats[tag]['attempted_recent'] += 1

    if lc_df is not None and not lc_df.empty:
        for _, row in lc_df.iterrows():
            slug = row.get("problem_id")
            tags = row.get("tags", [])
            tags = [str(t).strip().lower() for t in tags if t and str(t).strip()]
            
            for tag in tags:
                tag_stats[tag]['total'] += 1
                tag_stats[tag]['lc_problems'].append(slug)
                if slug and slug in solved_lc:
                    tag_stats[tag]['solved_recent'] += 1
                    tag_stats[tag]['attempted_recent'] += 1

    # ---------- TOTAL SOLVES CONTEXT ----------
    recent_solved = len(solved_cf) + len(solved_lc)
    total_solved = recent_solved + lc_total_solved
    
    cf_penalty = contest_penalty_cf(cf_contests) if cf_contests else 0.5
    lc_penalty = contest_penalty_lc(lc_contests) if lc_contests else 0.5
    combined_penalty = (cf_penalty + lc_penalty) / 2
    
    print(f"   🎯 Total context: recent={recent_solved}, total={total_solved}")
    print(f"   🎯 Contest penalty: {combined_penalty:.3f}")

    # ---------- EXPERT MODE LOGIC ----------
    if total_solved > 400:
        # EXPERTS: Focus on CONTEST-critical topics + relative performance
        adjusted_threshold = 0.50  # Need CLEAR weakness evidence
        adjusted_min = 8
        print(f"   🎯 EXPERT MODE: threshold=0.50, min_recent={adjusted_min}")
    else:
        adjusted_threshold = threshold + (combined_penalty * 0.1)
        adjusted_min = max(3, min_attempts)
        print(f"   🎯 Threshold: {adjusted_threshold:.3f}")

    # CONTEST-CRITICAL TOPICS (experts MUST master these)
    contest_critical = {
        'dp', 'dynamic programming', 'graphs', 'graph', 'trees', 'tree',
        'two pointers', 'binary search', 'dfs', 'bfs', 'greedy', 
        'backtracking', 'sliding window', 'heap', 'priority queue'
    }

    # ---------- PHASE 1: Statistical weaknesses ----------
    weak_candidates = []
    for tag, stats in tag_stats.items():
        if not tag or stats['attempted_recent'] < adjusted_min:
            continue
            
        success_rate = stats['solved_recent'] / max(stats['total'], 1)
        
        # Statistical weakness (low recent success)
        if success_rate < adjusted_threshold:
            contest_boost = 2.0 if tag in contest_critical else 1.0
            priority = stats['attempted_recent'] * (1 - success_rate) * contest_boost
            weak_candidates.append({
                'tag': tag, 'priority': priority, 'success_rate': success_rate,
                'solved': stats['solved_recent'], 'total': stats['total'],
                'type': 'statistical'
            })

    # ---------- PHASE 2: EXPERT STRATEGIC GAPS ----------
    strategic_gaps = []
    if total_solved > 400:
        for tag in contest_critical:
            if tag in tag_stats:
                stats = tag_stats[tag]
                if stats['attempted_recent'] < 3:  # Rarely practiced
                    strategic_gaps.append({
                        'tag': tag, 'priority': 10.0 - stats['attempted_recent'],
                        'success_rate': 0.0, 'solved': 0, 'total': stats['total'],
                        'type': 'strategic'
                    })

    # ---------- COMBINE & PRIORITIZE ----------
    all_weak = weak_candidates + strategic_gaps
    all_weak.sort(key=lambda x: (-x['priority'], x['success_rate']))
    
    weak_topics = [w['tag'] for w in all_weak[:10]]

    # ---------- EXPERT FALLBACK ----------
    if not weak_topics and total_solved > 200:
        weak_topics = sorted(contest_critical)[:8]
        print("   🎯 EXPERT: Contest preparation focus")

    # ---------- DEBUG ----------
    if all_weak:
        top = all_weak[0]
        print(f"   📊 Top: {top['tag']} ({top['solved']}/{top['total']} = {top['success_rate']:.0%})")
        print(f"   📊 Weak topics: {weak_topics[:6]}")
    elif total_solved > 50:
        print("   🎉 No statistical weaknesses - excellent!")

    return weak_topics
