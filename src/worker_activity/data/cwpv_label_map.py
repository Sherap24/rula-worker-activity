"""Load and apply CWPV motion label mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker_activity.config import find_repo_root, load_yaml


@dataclass(frozen=True)
class CwpvLabelMapping:
    motion_id: str
    raw_motion_label: str
    canonical: str
    confidence: str | None
    mapping_status: str
    include_in_baseline: bool
    exclusion_reason: str | None
    notes: str | None


def load_cwpv_label_map(path: Path | None = None) -> dict[str, CwpvLabelMapping]:
    root = find_repo_root()
    map_path = path or root / "configs" / "label_map_cwpv.yaml"
    data = load_yaml(map_path)
    result: dict[str, CwpvLabelMapping] = {}
    for item in data.get("mappings", []):
        motion_id = str(item["motion_id"]).strip()
        result[motion_id] = CwpvLabelMapping(
            motion_id=motion_id,
            raw_motion_label=str(item.get("raw_motion_label", "")),
            canonical=str(item["canonical"]),
            confidence=item.get("confidence"),
            mapping_status=str(item.get("mapping_status", "provisional")),
            include_in_baseline=bool(item.get("include_in_baseline", False)),
            exclusion_reason=_none_if_null(item.get("exclusion_reason")),
            notes=item.get("notes"),
        )
    return result


def _none_if_null(value: Any) -> str | None:
    if value is None or value == "null":
        return None
    return str(value)


def apply_cwpv_label_mapping(
    motion_id: str | None,
    label_map: dict[str, CwpvLabelMapping],
) -> dict[str, Any]:
    if motion_id is None or str(motion_id).strip() == "":
        return {
            "canonical_activity": None,
            "mapping_confidence": None,
            "mapping_status": "unmapped",
            "include_in_baseline": False,
            "exclusion_reason": "Missing motion_id",
        }
    key = str(motion_id).strip()
    mapping = label_map.get(key)
    if mapping is None:
        return {
            "canonical_activity": "unknown",
            "mapping_confidence": None,
            "mapping_status": "unmapped",
            "include_in_baseline": False,
            "exclusion_reason": f"No mapping for motion M{key}",
        }
    return {
        "canonical_activity": mapping.canonical,
        "mapping_confidence": mapping.confidence,
        "mapping_status": mapping.mapping_status,
        "include_in_baseline": mapping.include_in_baseline,
        "exclusion_reason": mapping.exclusion_reason,
    }
