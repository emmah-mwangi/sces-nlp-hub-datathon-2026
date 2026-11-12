"""
data_quality.py — Data quality checker for Team 7 (Datalynk) Lingala contribution dataset.

Flags issues for human review. Never deletes or rewrites records automatically.

Usage:
    python data_quality.py --data data/contribution.csv
    python data_quality.py --data data/contribution.csv --near-dup-threshold 0.85
"""
import argparse
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

VALID_LABELS = {
    "weather_query", "datetime_query", "alarm_set", "calendar_set",
    "calendar_query", "lists_createoradd", "lists_query",
    "play_music", "news_query", "qa_definition",
}
VALID_AUTHOR_CODES = {"T07-A", "T07-B", "T07-C", "T07-V1", "T07-V2", "T07-V3"}
VALID_REVIEW_STATUSES = {"self_reviewed", "independently_reviewed", "pending"}
VALID_SOURCE_METHODS = {
    "original_human", "human_localised",
    "newly_annotated", "corrected_with_evidence",
}
REQUIRED_COLUMNS = [
    "record_id", "language", "text", "label",
    "source_dataset", "source_version", "source_record_id", "source_url",
    "source_method", "licence_or_permission",
    "contributor_code", "reviewer_code", "review_status",
    "group_id", "split",
]
# Patterns that suggest personal information leaked into text
_PII_PATTERN = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"   # email
    r"|(?:\+254|0)[17]\d{8}"                  # Kenyan phone
    r"|\b\d{9,}\b",                            # long numeric string
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def check(path: str, near_dup_threshold: float = 0.85) -> dict:
    """
    Run all quality checks on the CSV at *path*.

    Returns a dict with keys:
        issues   — list of {record_id, check, detail} dicts (flag, not delete)
        summary  — human-readable string
        counts   — dict of check_name -> count
    """
    issues: list[dict] = []

    # ── 1. Load ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        return {
            "issues": [{"record_id": "N/A", "check": "load_error", "detail": str(exc)}],
            "summary": f"FATAL: could not load {path}: {exc}",
            "counts": {"load_error": 1},
        }

    # Legacy header compatibility (matches validate.py)
    if "label" not in df.columns and "intent" in df.columns:
        df = df.rename(columns={"intent": "label"})

    # ── 2. Missing columns ───────────────────────────────────────────────────
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append({
            "record_id": "N/A",
            "check": "missing_columns",
            "detail": "Missing columns: " + ", ".join(sorted(missing_cols)),
        })

    if df.empty:
        counts = _count(issues)
        return {"issues": issues, "summary": _summary(path, 0, issues, counts), "counts": counts}

    # ── 3. Per-row checks ────────────────────────────────────────────────────
    for _, row in df.iterrows():
        rid = row.get("record_id", "UNKNOWN")

        # Empty required fields — split and group_id are legitimately deferred
        DEFERRED = {"split", "group_id"}
        for col in REQUIRED_COLUMNS:
            if col in df.columns and col not in DEFERRED and not str(row.get(col, "")).strip():
                issues.append({"record_id": rid, "check": "empty_field", "detail": f"Column '{col}' is empty."})

        # Invalid label
        label = str(row.get("label", "")).strip()
        if label and label not in VALID_LABELS:
            issues.append({"record_id": rid, "check": "invalid_label", "detail": f"Unknown label: '{label}'"})

        # Invalid source_method
        method = str(row.get("source_method", "")).strip()
        if method and method not in VALID_SOURCE_METHODS and method != "practice_fixture":
            issues.append({"record_id": rid, "check": "invalid_source_method", "detail": f"Unexpected source_method: '{method}'"})

        # Invalid review_status
        rs = str(row.get("review_status", "")).strip()
        if rs and rs not in VALID_REVIEW_STATUSES:
            issues.append({"record_id": rid, "check": "invalid_review_status", "detail": f"Unexpected review_status: '{rs}'"})

        # Unrecognised author code
        cc = str(row.get("contributor_code", "")).strip()
        if cc and cc not in VALID_AUTHOR_CODES:
            issues.append({"record_id": rid, "check": "unrecognised_author", "detail": f"Unknown contributor_code: '{cc}'"})

        # Missing provenance (source_dataset blank)
        if not str(row.get("source_dataset", "")).strip():
            issues.append({"record_id": rid, "check": "missing_provenance", "detail": "source_dataset is blank."})

        # PII in text
        text = str(row.get("text", ""))
        if _PII_PATTERN.search(text):
            issues.append({"record_id": rid, "check": "pii_pattern", "detail": "Possible personal identifier in text field."})

        # Encoding: non-UTF-8-representable characters (pandas already decoded; flag replacement chars)
        if "\ufffd" in text:
            issues.append({"record_id": rid, "check": "encoding_problem", "detail": "Replacement character (U+FFFD) found — possible encoding issue."})

    # ── 4. Duplicate record IDs ──────────────────────────────────────────────
    dup_ids = df[df["record_id"].duplicated(keep=False)]["record_id"].unique()
    for rid in dup_ids:
        issues.append({"record_id": rid, "check": "duplicate_id", "detail": "record_id appears more than once."})

    # ── 5. Exact duplicate text ──────────────────────────────────────────────
    normed = df["text"].map(_normalise)
    dup_mask = normed.duplicated(keep=False)
    if dup_mask.any():
        dup_groups = df[dup_mask].groupby(normed[dup_mask])["record_id"].apply(list)
        for _, rids in dup_groups.items():
            issues.append({
                "record_id": ", ".join(rids),
                "check": "exact_duplicate_text",
                "detail": f"Identical text (after normalisation) shared by: {rids}",
            })

    # ── 6. Near-duplicate text ───────────────────────────────────────────────
    normed_list = normed.tolist()
    id_list = df["record_id"].tolist()
    for i in range(len(normed_list)):
        for j in range(i + 1, len(normed_list)):
            sim = _similarity(normed_list[i], normed_list[j])
            if sim >= near_dup_threshold:
                issues.append({
                    "record_id": f"{id_list[i]}, {id_list[j]}",
                    "check": "near_duplicate_text",
                    "detail": f"Similarity {sim:.2f} >= threshold {near_dup_threshold}: '{id_list[i]}' and '{id_list[j]}'",
                })

    counts = _count(issues)
    return {"issues": issues, "summary": _summary(path, len(df), issues, counts), "counts": counts}


def _count(issues: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for iss in issues:
        counts[iss["check"]] = counts.get(iss["check"], 0) + 1
    return counts


def _summary(path: str, n_records: int, issues: list[dict], counts: dict) -> str:
    lines = [
        f"Quality report for: {path}",
        f"Records loaded: {n_records}",
        f"Total flags: {len(issues)}",
    ]
    if counts:
        lines.append("Flags by type:")
        for check_name, cnt in sorted(counts.items()):
            lines.append(f"  {check_name}: {cnt}")
    else:
        lines.append("No issues found.")
    return "\n".join(lines)


def print_report(result: dict) -> None:
    print(result["summary"])
    if result["issues"]:
        print("\nDetailed flags (review required — no records deleted):")
        for iss in result["issues"]:
            print(f"  [{iss['check']}] record_id={iss['record_id']} — {iss['detail']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Data quality checker for Lingala contribution CSV.")
    p.add_argument("--data", default="data/contribution.csv")
    p.add_argument("--near-dup-threshold", type=float, default=0.85,
                   help="Similarity threshold for near-duplicate detection (0–1, default 0.85).")
    args = p.parse_args()
    result = check(args.data, args.near_dup_threshold)
    print_report(result)
