def normalize_cf(cf_df):
    cf = cf_df.copy()

    cf["difficulty"] = cf["rating"].fillna(1200)
    cf["tags"] = cf["tags"].apply(lambda x: x if isinstance(x, list) else [])

    cf["problem_id"] = cf["contestId"].astype(str) + cf["index"]
    cf["platform"] = "Codeforces"

    return cf[[
        "problem_id",
        "name",
        "difficulty",
        "tags",
        "link",
        "platform"
    ]]


def normalize_lc(lc_df):
    lc = lc_df.copy()

    diff_map = {
        "Easy": 800,
        "Medium": 1200,
        "Hard": 1600
    }

    lc["difficulty"] = lc["Difficulty Level"].map(diff_map)
    lc["tags"] = lc["tags"].apply(lambda x: x if isinstance(x, list) else [])

    lc["problem_id"] = lc["Question Slug"]
    lc["platform"] = "LeetCode"
    lc["name"] = lc["Question Title"]

    return lc[[
        "problem_id",
        "name",
        "difficulty",
        "tags",
        "link",
        "platform"
    ]]
