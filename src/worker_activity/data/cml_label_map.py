"""Load and apply CML label mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker_activity.config import find_repo_root, load_yaml


@dataclass(frozen=True)
class CmlLabelMapping:
    cml_label: str
    canonical: str
    confidence: str | None
    mapping_status: str
    include_in_baseline: bool
    exclusion_reason: str | None
    construction_subset_count_files: int | None
    notes: str | None


def load_cml_label_map(path: Path | None = None) -> dict[str, CmlLabelMapping]:
    root = find_repo_root()
    map_path = path or root / "configs" / "label_map_cml.yaml"
    data = load_yaml(map_path)
    result: dict[str, CmlLabelMapping] = {}
    for section in ("mappings",):
        for item in data.get(section, []):
            label = str(item["cml_label"]).strip().lower()
            result[label] = CmlLabelMapping(
                cml_label=str(item["cml_label"]),
                canonical=str(item["canonical"]),
                confidence=item.get("confidence"),
                mapping_status=str(item.get("mapping_status", "provisional")),
                include_in_baseline=bool(item.get("include_in_baseline", False)),
                exclusion_reason=_none_if_null(item.get("exclusion_reason")),
                construction_subset_count_files=item.get("construction_subset_count_files"),
                notes=item.get("notes"),
            )
    return result


def _none_if_null(value: Any) -> str | None:
    if value is None or value == "null":
        return None
    return str(value)


def apply_label_mapping(
    raw_label: str | None,
    label_map: dict[str, CmlLabelMapping],
) -> dict[str, Any]:
    """Return mapping fields for one raw CML label."""
    if raw_label is None or str(raw_label).strip() == "":
        return {
            "canonical_activity": None,
            "mapping_confidence": None,
            "mapping_status": "unmapped",
            "include_in_baseline": False,
            "exclusion_reason": "Missing raw activity label",
            "label_mapping_status": "unmapped",
        }

    key = str(raw_label).strip().lower()
    if key == "dynamic calibration":
        entry = label_map.get(key)
        if entry:
            return _mapping_to_row(entry)
        return {
            "canonical_activity": "unknown",
            "mapping_confidence": "high",
            "mapping_status": "calibration_only",
            "include_in_baseline": False,
            "exclusion_reason": "Calibration sequence",
            "label_mapping_status": "not_applicable",
        }

    entry = label_map.get(key)
    if entry is None:
        return {
            "canonical_activity": "unknown",
            "mapping_confidence": None,
            "mapping_status": "unmapped",
            "include_in_baseline": False,
            "exclusion_reason": f"No mapping defined for label '{raw_label}'",
            "label_mapping_status": "unmapped",
        }
    return _mapping_to_row(entry)


def _mapping_to_row(entry: CmlLabelMapping) -> dict[str, Any]:
    return {
        "canonical_activity": entry.canonical,
        "mapping_confidence": entry.confidence,
        "mapping_status": entry.mapping_status,
        "include_in_baseline": entry.include_in_baseline,
        "exclusion_reason": entry.exclusion_reason,
        "label_mapping_status": "mapped",
    }


def is_calibration_label(raw_label: str | None) -> bool:
    return str(raw_label or "").strip().lower() == "dynamic calibration"
