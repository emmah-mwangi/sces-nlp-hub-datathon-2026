"""
reconcile.py -- Reconciliation between contribution.csv and the private Record Log.

Usage:
    python reconcile.py
    python reconcile.py --contribution data/contribution.csv --log private/source_log/record_log.csv
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
DATASET_TARGET = 250
LABEL_TARGET = 25


def reconcile(contribution_path: str, log_path: str) -> str:
    lines = ["=" * 60, "RECONCILIATION REPORT -- Team 7 (Datalynk)", "=" * 60]

    contrib_path = Path(contribution_path)
    log_path_ = Path(log_path)

    if not contrib_path.exists():
        lines.append(f"FATAL: contribution file not found: {contrib_path}")
        return "\n".join(lines)

    contrib = pd.read_csv(contrib_path, dtype=str, keep_default_na=False)
    if "label" not in contrib.columns and "intent" in contrib.columns:
        contrib = contrib.rename(columns={"intent": "label"})

    has_log = log_path_.exists()
    log = pd.read_csv(log_path_, dtype=str, keep_default_na=False) if has_log else pd.DataFrame()

    contrib_ids = set(contrib["record_id"].tolist())
    log_ids = set(log["record_id"].tolist()) if has_log else set()

    # 1. Record counts
    lines.append("\n1. RECORD COUNTS")
    lines.append(f"   contribution.csv : {len(contrib)} records")
    lines.append(f"   record_log.csv   : {len(log)} records")
    shortfall = DATASET_TARGET - len(contrib)
    if shortfall > 0:
        lines.append(f"   SHORTFALL: {shortfall} records still needed to reach {DATASET_TARGET}.")
    else:
        lines.append(f"   OK: Dataset target of {DATASET_TARGET} reached.")

    # 2. CSV vs Log cross-check
    lines.append("\n2. CSV <-> LOG CROSS-CHECK")
    in_csv_not_log = contrib_ids - log_ids
    in_log_not_csv = log_ids - contrib_ids
    if in_csv_not_log:
        lines.append(f"   WARN: In CSV but missing from Record Log ({len(in_csv_not_log)}): {sorted(in_csv_not_log)}")
    else:
        lines.append("   OK: All CSV records have a Record Log entry.")
    if in_log_not_csv:
        lines.append(f"   WARN: In Record Log but missing from CSV ({len(in_log_not_csv)}): {sorted(in_log_not_csv)}")
    else:
        lines.append("   OK: All Record Log entries have a CSV record.")

    # 3. Duplicate IDs
    lines.append("\n3. DUPLICATE IDs")
    dup_csv = contrib[contrib["record_id"].duplicated(keep=False)]["record_id"].unique().tolist()
    if dup_csv:
        lines.append(f"   WARN: Duplicate IDs in CSV: {dup_csv}")
    else:
        lines.append("   OK: No duplicate IDs in CSV.")
    if has_log:
        dup_log = log[log["record_id"].duplicated(keep=False)]["record_id"].unique().tolist()
        if dup_log:
            lines.append(f"   WARN: Duplicate IDs in Record Log: {dup_log}")
        else:
            lines.append("   OK: No duplicate IDs in Record Log.")

    # 4. Label distribution
    lines.append(f"\n4. LABEL DISTRIBUTION (target: {LABEL_TARGET} per label)")
    if "label" in contrib.columns:
        counts = contrib["label"].value_counts().to_dict()
        for lbl in VALID_LABELS:
            n = counts.get(lbl, 0)
            status = "ok" if n >= LABEL_TARGET else f"need {LABEL_TARGET - n} more"
            lines.append(f"   {lbl:<22} {n:>3}  {status}")
        unknown = set(counts) - set(VALID_LABELS)
        if unknown:
            lines.append(f"   WARN: Unknown labels in CSV: {unknown}")

    # 5. Author attribution
    lines.append("\n5. AUTHOR ATTRIBUTION")
    missing_author = contrib[contrib["contributor_code"].str.strip() == ""]["record_id"].tolist()
    if missing_author:
        lines.append(f"   WARN: Records missing contributor_code ({len(missing_author)}): {missing_author}")
    else:
        lines.append("   OK: All records have a contributor_code.")

    # 6. Volunteer allocation
    lines.append("\n6. VOLUNTEER ALLOCATION (test records)")
    vol_records = contrib[contrib["contributor_code"].isin(VOLUNTEER_CODES)]
    for code, target in VOLUNTEER_TARGETS.items():
        n = (vol_records["contributor_code"] == code).sum()
        status = "ok" if n >= target else f"need {target - n} more"
        lines.append(f"   {code}: {n}/{target}  {status}")

    # 7. Test record review status
    lines.append("\n7. TEST RECORD REVIEW STATUS")
    if "split" in contrib.columns:
        test_records = contrib[contrib["split"] == "test"]
        if test_records.empty:
            lines.append("   INFO: No test split assigned yet (Tatyana's responsibility).")
        else:
            not_reviewed = test_records[test_records["review_status"] != "independently_reviewed"]
            if not_reviewed.empty:
                lines.append(f"   OK: All {len(test_records)} test records independently reviewed.")
            else:
                lines.append(
                    f"   WARN: {len(not_reviewed)} test records NOT independently reviewed: "
                    f"{not_reviewed['record_id'].tolist()}"
                )
            same = test_records[
                test_records["contributor_code"] == test_records["reviewer_code"]
            ]
            if not same.empty:
                lines.append(f"   WARN: Reviewer = author on test records: {same['record_id'].tolist()}")
    else:
        lines.append("   INFO: 'split' column not present -- splits not yet assigned.")

    # 8. Approval status
    if has_log and "approval_status" in log.columns:
        lines.append("\n8. APPROVAL STATUS (from Record Log)")
        pending = log[log["approval_status"] != "approved"]["record_id"].tolist()
        if pending:
            lines.append(f"   WARN: Records not yet approved ({len(pending)}): {pending}")
        else:
            lines.append("   OK: All logged records are approved.")
    else:
        lines.append("\n8. APPROVAL STATUS -- Record Log not available or missing column.")

    # 9. Split readiness
    lines.append("\n9. SPLIT READINESS (for Tatyana)")
    if "split" in contrib.columns:
        split_counts = contrib["split"].value_counts().to_dict()
        assigned = sum(v for k, v in split_counts.items() if k.strip())
        unassigned = split_counts.get("", 0)
        lines.append(f"   Assigned splits: {assigned}  |  Unassigned: {unassigned}")
        for sp in ["train", "dev", "test"]:
            lines.append(f"   {sp}: {split_counts.get(sp, 0)}")
        if unassigned:
            lines.append(
                f"   INFO: {unassigned} records have no split -- Tatyana must assign before final validation."
            )
    else:
        lines.append("   INFO: No 'split' column -- Tatyana must assign all splits.")

    # 10. Group ID integrity
    lines.append("\n10. GROUP ID INTEGRITY")
    if "group_id" in contrib.columns and "split" in contrib.columns:
        split_assigned = contrib[contrib["split"].str.strip() != ""]
        if not split_assigned.empty:
            cross = split_assigned.groupby("group_id")["split"].nunique()
            crossing = cross[cross > 1].index.tolist()
            if crossing:
                lines.append(f"   WARN: group_ids crossing splits (flag for Tatyana): {crossing}")
            else:
                lines.append("   OK: No group_id crosses multiple splits.")
        else:
            lines.append("   INFO: Splits not yet assigned -- group integrity check deferred.")
    else:
        lines.append("   INFO: group_id or split column missing -- check deferred.")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reconcile contribution.csv against the private Record Log.")
    p.add_argument("--contribution", default="data/contribution.csv")
    p.add_argument("--log", default="private/source_log/record_log.csv")
    args = p.parse_args()
    print(reconcile(args.contribution, args.log))
