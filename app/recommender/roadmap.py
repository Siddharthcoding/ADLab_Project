# recommender/roadmap.py
import math

LC_DIFF_MAP = {
    "easy": 800,
    "medium": 1200,
    "hard": 1600
}

def _normalize_lc_difficulty(val):
    if val is None:
        return 1200
    if isinstance(val, (int, float)):
        return int(val)
    v = str(val).strip().lower()
    return LC_DIFF_MAP.get(v, 1200)

def _normalize_cf_difficulty(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 1200
    try:
        return int(val)
    except Exception:
        return 1200

def _extract_lc_id(row):
    return (
        row.get("problem_id")
        or row.get("titleSlug")
        or row.get("Question Slug")
        or row.get("slug")
        or None
    )

def _extract_lc_title(row):
    return (
        row.get("name")
        or row.get("title")
        or row.get("Question Title")
        or row.get("questionTitle")
        or "Unknown"
    )

def _extract_tags(field):
    if field is None:
        return []
    if isinstance(field, list):
        out = []
        for t in field:
            if isinstance(t, dict):
                out.append(str(t.get("name", "")).strip().lower())
            else:
                out.append(str(t).strip().lower())
        return [x for x in out if x]
    fs = str(field)
    if "," in fs:
        return [x.strip().lower() for x in fs.split(",") if x.strip()]
    return [fs.strip().lower()] if fs.strip() else []

def generate_roadmap(
    lc_df,
    cf_df,
    weak_topics,
    solved_lc,
    solved_cf,
    contest_penalty,
    max_items=50,
    balance_platforms=True
):
    """
    Build a prioritized roadmap with adaptive platform balancing.
    If user is only active on one platform, prioritize that platform.
    """
    weak_set = set(str(t).lower().strip() for t in (weak_topics or []) if t)
    
    lc_pool = []
    cf_pool = []
    
    # Determine user's platform preference
    lc_solved_count = len(solved_lc)
    cf_solved_count = len(solved_cf)
    
    # If one platform has zero activity, focus on the active one
    if lc_solved_count == 0 and cf_solved_count > 0:
        lc_weight_multiplier = 0.3  # Deprioritize LC
        cf_weight_multiplier = 1.5  # Prioritize CF
    elif cf_solved_count == 0 and lc_solved_count > 0:
        lc_weight_multiplier = 1.5  # Prioritize LC
        cf_weight_multiplier = 0.3  # Deprioritize CF
    else:
        lc_weight_multiplier = 1.0
        cf_weight_multiplier = 1.0
    
    # Set difficulty limit based on contest performance
    max_diff = int(800 + (1-contest_penalty) * 1400)
    if contest_penalty < 0.3:  # Strong performer
        max_diff += 400
    
    # ---------- LeetCode Problems ----------
    if lc_df is not None and not lc_df.empty:
        for _, row in lc_df.iterrows():
            lid = _extract_lc_id(row)
            
            if lid and lid in solved_lc:
                continue
            
            tags = _extract_tags(
                row.get("tags") or 
                row.get("Topic Tagged text") or 
                row.get("topicTags")
            )
            
            diff = _normalize_lc_difficulty(
                row.get("difficulty") or 
                row.get("Difficulty Level")
            )
            
            if diff > max_diff:
                continue
            
            matches = bool(weak_set.intersection(set(tags)))
            
            weight = 1.0 * lc_weight_multiplier
            if matches:
                weight += 2.0
            weight += (max_diff - diff) / 1000
            
            lc_pool.append({
                "source": "LeetCode",
                "id": lid or f"lc_{row.name}",
                "name": _extract_lc_title(row),
                "difficulty": diff,
                "tags": tags,
                "matches": matches,
                "weight": weight,
                "link": row.get("link", "")
            })
    
    # ---------- Codeforces Problems ----------
    if cf_df is not None and not cf_df.empty:
        for _, row in cf_df.iterrows():
            pid = row.get("problem_id")
            
            if pid and pid in solved_cf:
                continue
            
            tags = _extract_tags(row.get("tags"))
            diff = _normalize_cf_difficulty(row.get("difficulty") or row.get("rating"))
            
            if diff > max_diff:
                continue
            
            matches = bool(weak_set.intersection(set(tags)))
            
            weight = 1.0 * cf_weight_multiplier
            if matches:
                weight += 2.0
            weight += (max_diff - diff) / 1000
            
            cf_pool.append({
                "source": "Codeforces",
                "id": pid or f"cf_{row.name}",
                "name": row.get("name", "Unknown"),
                "difficulty": diff,
                "tags": tags,
                "matches": matches,
                "weight": weight,
                "link": row.get("link", "")
            })
    
    # ---------- Sort Each Platform's Pool ----------
    lc_pool.sort(key=lambda x: (0 if x["matches"] else 1, -x["weight"], x["difficulty"]))
    cf_pool.sort(key=lambda x: (0 if x["matches"] else 1, -x["weight"], x["difficulty"]))
    
    # ---------- Create Roadmap Based on Platform Activity ----------
    final_roadmap = []
    
    if lc_solved_count == 0 and cf_solved_count > 0:
        # User only on CF: 70% CF, 30% LC
        cf_count = int(max_items * 0.7)
        lc_count = max_items - cf_count
        final_roadmap = cf_pool[:cf_count] + lc_pool[:lc_count]
        
    elif cf_solved_count == 0 and lc_solved_count > 0:
        # User only on LC: 70% LC, 30% CF
        lc_count = int(max_items * 0.7)
        cf_count = max_items - lc_count
        final_roadmap = lc_pool[:lc_count] + cf_pool[:cf_count]
        
    elif balance_platforms:
        # Active on both: 50-50 split
        lc_idx, cf_idx = 0, 0
        
        while len(final_roadmap) < max_items and (lc_idx < len(lc_pool) or cf_idx < len(cf_pool)):
            if lc_idx < len(lc_pool):
                final_roadmap.append(lc_pool[lc_idx])
                lc_idx += 1
            
            if len(final_roadmap) < max_items and cf_idx < len(cf_pool):
                final_roadmap.append(cf_pool[cf_idx])
                cf_idx += 1
    else:
        # Combine and sort by weight
        combined = lc_pool + cf_pool
        combined.sort(key=lambda x: (0 if x["matches"] else 1, -x["weight"], x["difficulty"]))
        final_roadmap = combined[:max_items]
    
    return final_roadmap
