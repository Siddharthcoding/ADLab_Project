def contest_penalty_cf(contests):
    if not contests:
        return 0.5
    drops = sum(1 for c in contests if c.get("newRating", 0) < c.get("oldRating", 1500))
    return drops / len(contests)

def contest_penalty_lc(history):
    if not history or len(history) < 2:
        return 0.5
    drops = sum(1 for i in range(1, len(history)) 
                if history[i].get("rating", 0) < history[i-1].get("rating", 1200))
    return drops / (len(history) - 1)
