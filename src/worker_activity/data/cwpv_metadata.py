"""CWPV video filename metadata helpers (no pose estimation)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# README example: 0931Camera_3.avi — participant 09, motion 3, trial 1, camera 3
# Extracted archives often use a 5-digit prefix with a block/session digit:
# 01112Camera_1.avi — participant 01, motion 1, block 1, trial 2, camera 1
# 01121Camera_1.avi — participant 01, motion 1, block 2, trial 1, camera 1
_CWPV_FILENAME_RE_5 = re.compile(
    r"^(?P<participant>\d{2})(?P<motion>\d)(?P<block>\d)(?P<trial>\d)Camera_(?P<camera>\d)\.(?P<ext>avi|mp4|mov|mkv)$",
    re.IGNORECASE,
)
_CWPV_FILENAME_RE_4 = re.compile(
    r"^(?P<participant>\d{2})(?P<motion>\d)(?P<trial>\d)Camera_(?P<camera>\d)\.(?P<ext>avi|mp4|mov|mkv)$",
    re.IGNORECASE,
)

CWPV_MOTION_LABELS: dict[str, str] = {
    "1": "M1_overhead_hammer_standing",
    "2": "M2_overhead_hammer_lean_twist",
    "3": "M3_shovel_semi_squat",
    "4": "M4_squat_lift_hold",
    "5": "M5_squat_hammer",
    "6": "M6_kneel_hammer",
    "7": "M7_squat_carry_bricks",
    "8": "M8_lean_carry_bricks",
}


@dataclass(frozen=True)
class CwpvFilenameFields:
    participant_id: str
    motion_id: str
    trial_id: str
    camera_id: str
    raw_motion_label: str
    block_id: str | None = None


def parse_cwpv_filename(file_name: str) -> CwpvFilenameFields | None:
    """Parse CWPV video filename per dataset README and extracted-archive variants."""
    match = _CWPV_FILENAME_RE_5.match(file_name) or _CWPV_FILENAME_RE_4.match(file_name)
    if not match:
        return None
    motion_id = match.group("motion")
    groups = match.groupdict()
    return CwpvFilenameFields(
        participant_id=match.group("participant"),
        motion_id=motion_id,
        trial_id=match.group("trial"),
        camera_id=match.group("camera"),
        raw_motion_label=CWPV_MOTION_LABELS.get(motion_id, f"M{motion_id}"),
        block_id=groups.get("block"),
    )


def enrich_cwpv_row(row: dict[str, Any], file_path: Path) -> dict[str, Any]:
    """Add CWPV-specific fields to an inventory row."""
    parsed = parse_cwpv_filename(file_path.name)
    if parsed is None:
        row["metadata_status"] = "extracted"
        row["label_mapping_status"] = "pending_inspection"
        row["notes"] = _append_note(row.get("notes"), "cwpv_filename_unparsed")
        return row

    row["subject_id"] = parsed.participant_id
    row["view_id"] = f"camera_{parsed.camera_id}"
    row["repetition_id"] = parsed.trial_id
    row["clip_id"] = file_path.stem
    if parsed.block_id is not None:
        row["video_id"] = (
            f"P{parsed.participant_id}_M{parsed.motion_id}_"
            f"B{parsed.block_id}_T{parsed.trial_id}"
        )
    else:
        row["video_id"] = (
            f"P{parsed.participant_id}_M{parsed.motion_id}_T{parsed.trial_id}"
        )
    row["raw_activity_label"] = parsed.raw_motion_label
    row["metadata_status"] = "extracted"
    row["label_mapping_status"] = "pending_inspection"
    block_note = f"; block={parsed.block_id}" if parsed.block_id is not None else ""
    row["notes"] = _append_note(
        row.get("notes"),
        f"motion=M{parsed.motion_id}{block_note}; trial={parsed.trial_id}; camera={parsed.camera_id}",
    )
    return row


def _append_note(existing: str | None, fragment: str) -> str:
    if not existing:
        return fragment
    if fragment in existing:
        return existing
    return f"{existing}; {fragment}"
