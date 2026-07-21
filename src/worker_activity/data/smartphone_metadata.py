"""Local smartphone video filename helpers (Week 5 domain transfer)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Example: squatting_01.mp4, overhead_work_reaching_02.MP4
_SMARTPHONE_FILENAME_RE = re.compile(
    r"^(?P<label>[a-z][a-z0-9_]*)_(?P<clip>\d{2})\.(?P<ext>mp4|mov|avi|mkv)$",
    re.IGNORECASE,
)

CANONICAL_SMARTPHONE_ACTIVITIES = frozenset(
    {
        "carrying",
        "kneeling",
        "lifting_lowering",
        "overhead_work_reaching",
        "squatting",
    }
)


@dataclass(frozen=True)
class SmartphoneFilenameFields:
    canonical_activity: str
    clip_id: str
    raw_activity_label: str


def parse_smartphone_filename(file_name: str) -> SmartphoneFilenameFields | None:
    """Parse `{canonical_activity}_{nn}.ext` smartphone capture filenames."""
    match = _SMARTPHONE_FILENAME_RE.match(file_name)
    if not match:
        return None
    label = match.group("label").lower()
    if label not in CANONICAL_SMARTPHONE_ACTIVITIES:
        return None
    clip = match.group("clip")
    return SmartphoneFilenameFields(
        canonical_activity=label,
        clip_id=clip,
        raw_activity_label=label,
    )


def enrich_smartphone_row(row: dict[str, Any], file_path: Path) -> dict[str, Any]:
    """Add smartphone-specific label fields to an inventory row."""
    parsed = parse_smartphone_filename(file_path.name)
    if parsed is None:
        row["metadata_status"] = row.get("metadata_status") or "extracted"
        row["label_mapping_status"] = "unmapped"
        row["notes"] = _append_note(row.get("notes"), "smartphone_filename_unparsed")
        return row

    logical_id = f"phone_{parsed.canonical_activity}_{parsed.clip_id}"
    row["subject_id"] = "phone_self"
    row["clip_id"] = file_path.stem
    row["video_id"] = logical_id
    row["raw_activity_label"] = parsed.raw_activity_label
    row["canonical_activity"] = parsed.canonical_activity
    row["mapping_confidence"] = "high"
    row["mapping_status"] = "filename_label"
    row["label_mapping_status"] = "mapped"
    row["include_in_baseline"] = False
    row["exclusion_reason"] = "domain_transfer_eval_only"
    row["metadata_status"] = row.get("metadata_status") or "extracted"
    row["notes"] = _append_note(
        row.get("notes"),
        f"smartphone; activity={parsed.canonical_activity}; clip={parsed.clip_id}",
    )
    return row


def _append_note(existing: str | None, fragment: str) -> str:
    if not existing:
        return fragment
    if fragment in existing:
        return existing
    return f"{existing}; {fragment}"
