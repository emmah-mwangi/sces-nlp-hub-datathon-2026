"""
ingest_batch.py — Batch ingestion for Team 7 (Datalynk) Lingala contribution dataset.

Receives a batch CSV, validates format, assigns record IDs, detects duplicates,
appends clean records to data/contribution.csv, and updates the private Record Log.

Flagged records are written to a separate review file — never silently dropped.

Usage:
    python ingest_batch.py --batch path/to/batch.csv --batch-id batch-01 --date 2026-11-05
    python ingest_batch.py --batch path/to/batch.csv --batch-id batch-02 --date 2026-11-06 --dry-run
"""
import argparse
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

CONTRIBUTION_CSV = Path("data/contribution.csv")
RECORD_LOG_CSV = Path("private/source_log/record_log.csv")
REVIEW_DIR = Path("data/review")

VALID_LABELS = {
    "weather_query", "datetime_query", "alarm_set", "calendar_set",
    "calendar_query", "lists_createoradd", "lists_query",
    "play_music", "news_query", "qa_definition",
}
VALID_AUTHOR_CODES = {"T07-A", "T07-B", "T07-C", "T07-V1", "T07-V2", "T07-V3"}
VALID_SOURCE_METHODS = {
    "original_human", "human_localised",
    "newly_annotated", "corrected_with_evidence",
}
CONTRIBUTION_COLUMNS = [
    "record_id", "language", "text", "label",
    "source_dataset", "source_version", "source_record_id", "source_url",
    "source_method", "licence_or_permission",
    "contributor_code", "reviewer_code", "review_status",
    "group_id", "split",
]
LOG_COLUMNS = [
    "record_id", "contributor_code", "source_type", "batch",
    "date_received", "assigned_label", "review_status",
    "correction_status", "approval_status", "notes",
]
# Patterns that suggest personal information
_PII = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?:\+254|0)[17]\d{8}|\b\d{9,}\b",
    re.IGNORECASE,
)
ID_PREFIX = "T07"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip().casefold())


def _next_id(existing_ids: set[str]) -> callable:
    """Return a counter function that yields the next unused T07-NNN id."""
    used = set()
    for rid in existing_ids:
        m = re.match(rf"^{ID_PREFIX}-(\d+)$", rid)
        if m:
            used.add(int(m.group(1)))
    counter = [max(used, default=0)]

    def _next() -> str:
        counter[0] += 1
        while counter[0] in used:
            counter[0] += 1
        used.add(counter[0])
        return f"{ID_PREFIX}-{counter[0]:03d}"

    return _next


def _load_existing() -> tuple[pd.DataFrame, set[str], set[str]]:
    """Load contribution.csv; return (df, existing_ids, existing_normed_texts)."""
    if CONTRIBUTION_CSV.exists():
        df = pd.read_csv(CONTRIBUTION_CSV, dtype=str, keep_default_na=False)
        if "label" not in df.columns and "intent" in df.columns:
            df = df.rename(columns={"intent": "label"})
    else:
        df = pd.DataFrame(columns=CONTRIBUTION_COLUMNS)
    existing_ids = set(df["record_id"].tolist())
    existing_normed = set(df["text"].map(_normalise).tolist()) if not df.empty else set()
    return df, existing_ids, existing_normed


def _load_log() -> pd.DataFrame:
    if RECORD_LOG_CSV.exists():
        return pd.read_csv(RECORD_LOG_CSV, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=LOG_COLUMNS)


def ingest(batch_path: str, batch_id: str, received_date: str, dry_run: bool = False) -> dict:
    """
    Ingest a batch CSV.

    Returns a summary dict with keys: accepted, flagged, skipped, report.
    """
    batch_path = Path(batch_path)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    incoming = pd.read_csv(batch_path, dtype=str, keep_default_na=False)
    # Legacy header
    if "label" not in incoming.columns and "intent" in incoming.columns:
        incoming = incoming.rename(columns={"intent": "label"})

    existing_df, existing_ids, existing_normed = _load_existing()
    next_id = _next_id(existing_ids)

    accepted_rows: list[dict] = []
    log_rows: list[dict] = []
    flagged_rows: list[dict] = []
    report_lines: list[str] = []

    # Track normed texts seen in this batch to catch intra-batch duplicates
    batch_normed: set[str] = set()

    for idx, row in incoming.iterrows():
        row_num = idx + 2  # 1-based, accounting for header
        flags: list[str] = []

        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()
        contributor = str(row.get("contributor_code", "")).strip()
        source_method = str(row.get("source_method", "")).strip()
        language = str(row.get("language", "")).strip()

        # ── Structural checks ────────────────────────────────────────────────
        if not text:
            flags.append("empty_text")
        if not label:
            flags.append("empty_label")
        elif label not in VALID_LABELS:
            flags.append(f"invalid_label:{label}")
        if not contributor:
            flags.append("missing_contributor_code")
        elif contributor not in VALID_AUTHOR_CODES:
            flags.append(f"unrecognised_author:{contributor}")
        if source_method and source_method not in VALID_SOURCE_METHODS:
            flags.append(f"invalid_source_method:{source_method}")
        if not language:
            flags.append("missing_language")

        # ── PII check ────────────────────────────────────────────────────────
        if text and _PII.search(text):
            flags.append("pii_pattern")

        # ── Duplicate checks ─────────────────────────────────────────────────
        normed = _normalise(text) if text else ""
        if normed and normed in existing_normed:
            flags.append("exact_duplicate_existing")
        if normed and normed in batch_normed:
            flags.append("exact_duplicate_intra_batch")

        # ── Assign or verify record ID ───────────────────────────────────────
        incoming_rid = str(row.get("record_id", "")).strip()
        if incoming_rid and incoming_rid in existing_ids:
            flags.append(f"duplicate_id:{incoming_rid}")
            rid = incoming_rid  # keep for flagged log
        elif incoming_rid and re.match(rf"^{ID_PREFIX}-\d+$", incoming_rid):
            rid = incoming_rid  # accept supplied ID
        else:
            rid = next_id()  # assign new ID

        # ── Route record ─────────────────────────────────────────────────────
        if flags:
            flagged_rows.append({**row.to_dict(), "record_id": rid, "_flags": "; ".join(flags), "_row": row_num})
            report_lines.append(f"  ROW {row_num} [{rid}] FLAGGED: {'; '.join(flags)}")
        else:
            # Build full contribution row with defaults for optional fields
            contrib_row = {
                "record_id": rid,
                "language": language,
                "text": text,
                "label": label,
                "source_dataset": row.get("source_dataset", "team07_lingala_v1"),
                "source_version": row.get("source_version", "1.0"),
                "source_record_id": row.get("source_record_id", rid),
                "source_url": row.get("source_url", "not_applicable"),
                "source_method": source_method or "original_human",
                "licence_or_permission": row.get("licence_or_permission", "team_permission"),
                "contributor_code": contributor,
                "reviewer_code": row.get("reviewer_code", ""),
                "review_status": row.get("review_status", "pending"),
                "group_id": str(row.get("group_id", "")).strip() or rid,
                "split": str(row.get("split", "")).strip(),  # left blank — Tatyana assigns splits
            }
            log_row = {
                "record_id": rid,
                "contributor_code": contributor,
                "source_type": source_method or "original_human",
                "batch": batch_id,
                "date_received": received_date,
                "assigned_label": label,
                "review_status": row.get("review_status", "pending"),
                "correction_status": "none",
                "approval_status": "pending",
                "notes": "",
            }
            accepted_rows.append(contrib_row)
            log_rows.append(log_row)
            existing_ids.add(rid)
            existing_normed.add(normed)
            batch_normed.add(normed)
            report_lines.append(f"  ROW {row_num} [{rid}] accepted — label={label}, author={contributor}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    if not dry_run:
        if accepted_rows:
            new_df = pd.DataFrame(accepted_rows, columns=CONTRIBUTION_COLUMNS)
            updated = pd.concat([existing_df, new_df], ignore_index=True)
            updated.to_csv(CONTRIBUTION_CSV, index=False)

            log_df = _load_log()
            new_log = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
            updated_log = pd.concat([log_df, new_log], ignore_index=True)
            RECORD_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
            updated_log.to_csv(RECORD_LOG_CSV, index=False)

        if flagged_rows:
            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            flagged_path = REVIEW_DIR / f"{batch_id}_flagged.csv"
            pd.DataFrame(flagged_rows).to_csv(flagged_path, index=False)
            report_lines.append(f"\nFlagged records written to: {flagged_path}")
            report_lines.append("Review and correct flagged records, then re-ingest or manually add to contribution.csv.")

    summary = {
        "batch_id": batch_id,
        "date": received_date,
        "incoming": len(incoming),
        "accepted": len(accepted_rows),
        "flagged": len(flagged_rows),
        "dry_run": dry_run,
    }
    header = (
        f"\n{'[DRY RUN] ' if dry_run else ''}Batch ingestion report — {batch_id} ({received_date})\n"
        f"  Incoming: {summary['incoming']}  |  Accepted: {summary['accepted']}  |  Flagged: {summary['flagged']}\n"
    )
    summary["report"] = header + "\n".join(report_lines)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest a contribution batch into data/contribution.csv.")
    p.add_argument("--batch", required=True, help="Path to incoming batch CSV.")
    p.add_argument("--batch-id", required=True, help="Batch identifier, e.g. batch-01.")
    p.add_argument("--date", default=str(date.today()), help="Date received (YYYY-MM-DD). Defaults to today.")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing any files.")
    args = p.parse_args()
    result = ingest(args.batch, args.batch_id, args.date, args.dry_run)
    print(result["report"])
