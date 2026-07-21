"""CML skeleton file metadata helpers (no pose estimation)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_FOLDER_LABEL_RE = re.compile(r"^\d+(.+)$")
_JSON_STRING_FIELD_RE = re.compile(r'"([^"]+)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
_JSON_INT_FIELD_RE = re.compile(r'"([^"]+)"\s*:\s*(-?\d+)')


def parse_cml_folder_label(folder_name: str) -> str:
    """Parse activity name from folder like ``1bending`` → ``bending``."""
    match = _FOLDER_LABEL_RE.match(folder_name)
    return match.group(1) if match else folder_name


def infer_cml_subset(relative_path: str) -> str | None:
    if "Construction_Related_Data" in relative_path:
        return "construction_related"
    if "All_DATA" in relative_path:
        return "all_data"
    return None


def infer_joint_schema(relative_path: str) -> str | None:
    if "15_nodes" in relative_path:
        return "15_nodes"
    if "20_nodes" in relative_path:
        return "20_nodes"
    return None


def _fast_read_json_fields(path: Path, max_bytes: int = 8192) -> dict[str, Any]:
    """Read leading JSON scalar fields without parsing coordinate arrays."""
    try:
        with path.open(encoding="utf-8") as handle:
            chunk = handle.read(max_bytes)
    except OSError:
        return {}
    fields: dict[str, Any] = {}
    for key, value in _JSON_STRING_FIELD_RE.findall(chunk):
        if key not in fields:
            fields[key] = value.replace("\\/", "/")
    for key, value in _JSON_INT_FIELD_RE.findall(chunk):
        if key not in fields:
            fields[key] = int(value)
    return fields


def read_cml_json_record(path: Path) -> dict[str, Any] | None:
    """Read CML JSON header fields."""
    fields = _fast_read_json_fields(path)
    if not fields:
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        fields = data

    if "label" not in fields and "frames" not in fields:
        return None

    return {
        "raw_activity_label": fields.get("label"),
        "original_label": fields.get("original label"),
        "action_type": fields.get("action type"),
        "frame_count": fields.get("frames"),
        "data_source": fields.get("data source"),
        "source_file": fields.get("source file"),
        "joint_count": fields.get("joints"),
        "calibrated": fields.get("calibrated"),
    }


def enrich_cml_row(row: dict[str, Any], file_path: Path, relative_path: str) -> dict[str, Any]:
    """Add CML-specific fields to an inventory row."""
    parent = file_path.parent.name
    folder_label = parse_cml_folder_label(parent)
    subset = infer_cml_subset(relative_path)
    joint_schema = infer_joint_schema(relative_path)

    row["clip_id"] = file_path.stem
    row["raw_activity_label"] = folder_label
    row["view_id"] = joint_schema
    row["metadata_status"] = "extracted"
    row["label_mapping_status"] = "pending_inspection"

    notes_parts = []
    if subset:
        notes_parts.append(f"subset={subset}")
    if joint_schema:
        notes_parts.append(f"joints={joint_schema}")

    if file_path.suffix.lower() == ".json":
        record = read_cml_json_record(file_path)
        if record:
            json_label = record.get("raw_activity_label")
            if json_label and json_label != folder_label:
                notes_parts.append(f"json_label={json_label}")
                row["raw_activity_label"] = str(json_label)
            if record.get("original_label"):
                notes_parts.append(f"original_label={record['original_label']}")
            if record.get("action_type") is not None:
                notes_parts.append(f"action_type={record['action_type']}")
            if record.get("frame_count") is not None:
                row["frame_count"] = int(record["frame_count"])
            if record.get("data_source"):
                notes_parts.append(f"data_source={record['data_source']}")
            if record.get("source_file"):
                row["video_id"] = str(record["source_file"])
        else:
            row["metadata_status"] = "failed"
            notes_parts.append("json_parse_failed")

    row["notes"] = "; ".join(notes_parts) if notes_parts else None
    return row
