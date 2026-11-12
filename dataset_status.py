"""
dataset_status.py -- Current dataset status report for Team 7 (Datalynk).

Usage:
    python dataset_status.py
    python dataset_status.py --contribution data/contribution.csv
"""
import argparse
from pathlib import Path

import pandas as pd

VALID_LABELS = [
    "weather_query", "datetime_query", "alarm_set", "calendar_set",
    "calendar_query", "lists_createoradd", "lists_query",
    "play_music", "news_query", "qa_definition",
]
VOLUNTEER_CODES = {"T07-V1", "T07-V2", "T07-V3"}
VOLUNTEER_TARGETS = {"T07-V1": 17, "T07-V2": 17, "T07-V3": 16}
BATCH_MILESTONES = [
    ("BATCH 1 (Thu 5 Nov)", 80),
    ("BATCH 2 (Fri 6 Nov)", 150),
    ("BATCH 3 (Sat 7 Nov)", 250),
]
LABEL_TARGET = 25


def status_report(contribution_path: str) -> str:
    lines = ["=" * 60, "DATASET STATUS -- Team 7 (Datalynk) Lingala", "=" * 60]

    path = Path(contribution_path)
    if not path.exists():
        lines.append(f"contribution.csv not found at {path}. No records yet.")
        return "\n".join(lines)

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "label" not in df.columns and "intent" in df.columns:
        df = df.rename(columns={"intent": "label"})

    total = len(df)
    lines.append(f"\nTotal records: {total}")

    lines.append("\nBATCH MILESTONES:")
    for name, target in BATCH_MILESTONES:
        pct = min(100, round(total / target * 100)) if target else 0
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        status = "REACHED" if total >= target else f"need {target - total} more"
        lines.append(f"  {name:<28} {total:>3}/{target}  [{bar}]  {status}")

    lines.append(f"\nLABEL DISTRIBUTION (target: {LABEL_TARGET} each):")
    label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    for lbl in VALID_LABELS:
        n = label_counts.get(lbl, 0)
        bar = "#" * min(n, LABEL_TARGET) + "-" * max(0, LABEL_TARGET - n)
        status = "ok" if n >= LABEL_TARGET else f"need {LABEL_TARGET - n}"
        lines.append(f"  {lbl:<22} {n:>3}/{LABEL_TARGET}  [{bar}]  {status}")
    unknown_labels = set(label_counts) - set(VALID_LABELS)
    if unknown_labels:
        lines.append(f"  WARN: Unknown labels: {unknown_labels}")

    lines.append("\nVOLUNTEER ALLOCATION (test records):")
    for code, target in VOLUNTEER_TARGETS.items():
        n = (df["contributor_code"] == code).sum() if "contributor_code" in df.columns else 0
        status = "ok" if n >= target else f"need {target - n}"
        lines.append(f"  {code}: {n}/{target}  {status}")

    lines.append("\nREVIEW COVERAGE:")
    if "review_status" in df.columns:
        rs = df["review_status"].value_counts().to_dict()
        for k, v in sorted(rs.items()):
            lines.append(f"  {k}: {v}")
        unreviewed = df[~df["review_status"].isin(
            ["self_reviewed", "independently_reviewed"])]["record_id"].tolist()
        if unreviewed:
            lines.append(f"  WARN: Pending review ({len(unreviewed)}): {unreviewed}")
    else:
        lines.append("  review_status column not present.")

    lines.append("\nSPLIT READINESS:")
    if "split" in df.columns:
        sc = df["split"].value_counts().to_dict()
        for sp in ["train", "dev", "test", ""]:
            label = sp if sp else "(unassigned)"
            lines.append(f"  {label}: {sc.get(sp, 0)}")
    else:
        lines.append("  No split column -- Tatyana must assign splits.")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Print current dataset status.")
    p.add_argument("--contribution", default="data/contribution.csv")
    args = p.parse_args()
    print(status_report(args.contribution))
