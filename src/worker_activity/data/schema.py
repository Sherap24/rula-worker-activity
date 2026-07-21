"""Canonical clip inventory schema for Week 2."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Column order for clip inventory CSV/Parquet exports
CLIP_INVENTORY_COLUMNS: list[str] = [
    "source",
    "source_version",
    "source_type",
    "relative_path",
    "file_name",
    "extension",
    "size_bytes",
    "checksum",
    "video_id",
    "clip_id",
    "subject_id",
    "view_id",
    "repetition_id",
    "raw_activity_label",
    "canonical_activity",
    "label_mapping_status",
    "fps",
    "frame_count",
    "duration_seconds",
    "width",
    "height",
    "codec",
    "metadata_status",
    "integrity_status",
    "notes",
    # CML enrichment columns (nullable for non-CML sources)
    "mapping_confidence",
    "mapping_status",
    "include_in_baseline",
    "exclusion_reason",
    "subject_parse_status",
    "source_file",
    "source_dataset",
    "skeleton_layout",
    "logical_sample_id",
    "representation_group_id",
    "construction_subset",
]

REQUIRED_STRING_COLUMNS = {"source", "source_type", "relative_path", "file_name", "extension"}

NULLABLE_NUMERIC_COLUMNS = {
    "size_bytes",
    "fps",
    "frame_count",
    "duration_seconds",
    "width",
    "height",
}

VALID_METADATA_STATUS = {"pending", "extracted", "failed", "not_applicable"}
VALID_INTEGRITY_STATUS = {"pending", "ok", "failed", "skipped"}
VALID_LABEL_MAPPING_STATUS = {"pending_inspection", "mapped", "unmapped", "not_applicable"}


class SchemaValidationError(Exception):
    """Raised when inventory data fails schema validation."""


def empty_inventory_frame() -> pd.DataFrame:
    """Return an empty DataFrame with canonical columns and nullable dtypes."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in CLIP_INVENTORY_COLUMNS})


def validate_inventory_frame(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """
    Validate a clip inventory DataFrame.

    Returns a list of issues. Raises SchemaValidationError when *strict* and issues exist.
    """
    issues: list[str] = []

    missing_cols = [c for c in CLIP_INVENTORY_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in CLIP_INVENTORY_COLUMNS]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        issues.append(f"Unexpected columns: {extra_cols}")

    if df.empty:
        if strict and issues:
            raise SchemaValidationError("; ".join(issues))
        return issues

    for col in REQUIRED_STRING_COLUMNS:
        if col not in df.columns:
            continue
        nulls = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if nulls.any():
            issues.append(f"Column '{col}' has {int(nulls.sum())} null/empty values")

    if "metadata_status" in df.columns:
        bad = ~df["metadata_status"].isin(VALID_METADATA_STATUS | {None}) & df[
            "metadata_status"
        ].notna()
        if bad.any():
            issues.append("Invalid metadata_status values present")

    if "integrity_status" in df.columns:
        bad = ~df["integrity_status"].isin(VALID_INTEGRITY_STATUS | {None}) & df[
            "integrity_status"
        ].notna()
        if bad.any():
            issues.append("Invalid integrity_status values present")

    if strict and issues:
        raise SchemaValidationError("; ".join(issues))
    return issues


def normalize_inventory_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all canonical columns exist and are ordered."""
    out = empty_inventory_frame()
    if df.empty:
        return out
    for col in CLIP_INVENTORY_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = None
    return out[CLIP_INVENTORY_COLUMNS]


def inventory_row(
    *,
    source: str,
    source_type: str,
    relative_path: str,
    file_name: str,
    extension: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a single inventory row with defaults for nullable fields."""
    row = {col: None for col in CLIP_INVENTORY_COLUMNS}
    row.update(
        {
            "source": source,
            "source_type": source_type,
            "relative_path": relative_path,
            "file_name": file_name,
            "extension": extension,
            "metadata_status": "pending",
            "integrity_status": "pending",
            "label_mapping_status": "pending_inspection",
        }
    )
    row.update(kwargs)
    return row
