"""Tests for clip inventory schema."""

import pandas as pd
import pytest

from worker_activity.data.schema import (
    SchemaValidationError,
    empty_inventory_frame,
    inventory_row,
    normalize_inventory_frame,
    validate_inventory_frame,
)


def test_empty_inventory_valid():
    df = empty_inventory_frame()
    issues = validate_inventory_frame(df, strict=False)
    assert issues == []


def test_inventory_row_defaults():
    row = inventory_row(
        source="cwpv",
        source_type="video_and_imu",
        relative_path="cwpv/extracted/sample.mp4",
        file_name="sample.mp4",
        extension=".mp4",
    )
    assert row["metadata_status"] == "pending"
    assert row["label_mapping_status"] == "pending_inspection"


def test_normalize_adds_missing_columns():
    df = pd.DataFrame([{"source": "cml", "file_name": "a.csv"}])
    out = normalize_inventory_frame(df)
    assert "checksum" in out.columns
    assert len(out.columns) == len(empty_inventory_frame().columns)


def test_strict_validation_missing_columns():
    df = pd.DataFrame([{"source": "cwpv"}])
    with pytest.raises(SchemaValidationError):
        validate_inventory_frame(df, strict=True)
