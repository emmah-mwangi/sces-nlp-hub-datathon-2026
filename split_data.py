"""T23: propose group_id values, then split each intent into train/dev/test.

Related examples share a group_id and always land in the same split.
Test rows must be independently reviewed by someone other than the author.
Usage: python split_data.py --data data/contribution.csv
"""
import argparse
import random
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TARGET = {"test": 5, "dev": 5}  # train gets the rest (15 at the 25-per-intent minimum)
REQUIRED = ["record_id", "text", "label", "contributor_code", "reviewer_code",
            "review_status", "group_id", "split"]


def propose_groups(df, threshold):
    """Fill empty group_id by linking near-identical texts within one label."""
    df = df.copy()
    for label, g in df.groupby("label"):
        idx = list(g.index)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(g.text.str.lower())
        sim = cosine_similarity(vec)
        parent = list(range(len(idx)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if sim[i, j] >= threshold:
                    parent[find(i)] = find(j)
        for i, k in enumerate(idx):
            if not df.at[k, "group_id"].strip():
                df.at[k, "group_id"] = f"T07-{label}-g{find(i):03d}"
    return df


def test_eligible(sub):
    """True if every row in the group was independently reviewed by a different person."""
    return bool(((sub.review_status == "independently_reviewed")
                 & (sub.reviewer_code.str.strip() != "")
                 & (sub.reviewer_code != sub.contributor_code)).all())


def split_label(g, rng):
    groups = [(gid, list(sub.index), test_eligible(sub)) for gid, sub in g.groupby("group_id")]
    rng.shuffle(groups)
    assigned, used = {}, set()

    def take(split, need, only_eligible):
        for gid, idx, ok in groups:
            if need == 0:
                break
            if gid in used or (only_eligible and not ok) or len(idx) > need:
                continue
            for k in idx:
                assigned[k] = split
            used.add(gid)
            need -= len(idx)
        return need

    short_test = take("test", TARGET["test"], True)
    short_dev = take("dev", TARGET["dev"], False)
    for gid, idx, ok in groups:
        if gid not in used:
            for k in idx:
                assigned[k] = "train"
    return assigned, short_test, short_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contribution.csv")
    ap.add_argument("--out", default="data/contribution_split.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.data, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")
    if df.split.str.strip().ne("").any() and not a.force:
        sys.exit("Some rows already have a split. Splits are frozen. Use --force only deliberately.")
    df["split"] = ""

    df = propose_groups(df, a.threshold)
    rng = random.Random(a.seed)
    warnings = []
    for label, g in df.groupby("label"):
        assigned, st, sd = split_label(g, rng)
        for k, s in assigned.items():
            df.at[k, "split"] = s
        if st:
            warnings.append(f"{label}: {st} test row(s) short. Need more independently reviewed records.")
        if sd:
            warnings.append(f"{label}: {sd} dev row(s) short.")

    df.to_csv(a.out, index=False, encoding="utf-8")
    multi = df.groupby(["label", "group_id"]).filter(lambda x: len(x) > 1)
    multi.sort_values(["label", "group_id"])[["label", "group_id", "record_id", "text"]].to_csv(
        "data/group_review.csv", index=False, encoding="utf-8")

    print(pd.crosstab(df.label, df.split, margins=True))
    print(f"\nWrote {a.out} and data/group_review.csv ({multi.group_id.nunique()} multi-row groups to confirm).")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()