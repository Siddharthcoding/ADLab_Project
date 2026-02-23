# preprocess/codeforces_catalog.py

import pandas as pd
from app.api.codeforces_api import fetch_cf_problemset

def load_cf_catalog():
    """
    Load Codeforces problem catalog.
    Returns empty DataFrame if API fails.
    """
    problems = fetch_cf_problemset()
    
    if not problems:
        print("⚠️  Could not load Codeforces problemset. Using empty catalog.")
        return pd.DataFrame(columns=['name', 'rating', 'tags', 'contestId', 'index', 'link'])
    
    rows = []
    
    for p in problems:
        # Some problems might not have ratings
        rating = p.get("rating")
        contest_id = p.get("contestId")
        index = p.get("index")
        
        if contest_id and index:
            rows.append({
                "name": p.get("name", "Unknown"),
                "rating": rating,
                "tags": [t.lower() for t in p.get("tags", [])],
                "contestId": contest_id,
                "index": index,
                "link": f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
            })
    
    return pd.DataFrame(rows)
