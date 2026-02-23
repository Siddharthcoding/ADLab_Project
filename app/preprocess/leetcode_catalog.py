import pandas as pd

def load_leetcode_catalog():
    df = pd.read_csv("data/leetcode_questions.csv")

    df["tags"] = df["Topic Tagged text"].fillna("").apply(
        lambda x: [t.strip().lower() for t in x.split(",")]
    )

    df["difficulty"] = df["Difficulty Level"].map({
        "Easy": 800,
        "Medium": 1200,
        "Hard": 1600
    })

    df["link"] = "https://leetcode.com/problems/" + df["Question Slug"]

    return df
