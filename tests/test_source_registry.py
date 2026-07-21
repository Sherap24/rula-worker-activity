"""Tests for data source registry."""

from pathlib import Path

import pytest

from worker_activity.data.source_registry import (
    DataSource,
    SourceRegistryError,
    load_source_registry,
)


def test_load_source_registry(repo_root: Path):
    sources = load_source_registry(repo_root / "configs" / "data_sources.yaml")
    assert len(sources) >= 5
    ids = {s.id for s in sources}
    assert "cml" in ids
    assert "cwpv" in ids


def test_cml_local_status_after_download(repo_root: Path):
    sources = {s.id: s for s in load_source_registry(repo_root / "configs" / "data_sources.yaml")}
    assert sources["cml"].local_status == "present_unvalidated"
    assert sources["cwpv"].local_status == "present_unvalidated"


def test_cml_is_skeleton_not_video(repo_root: Path):
    sources = {s.id: s for s in load_source_registry(repo_root / "configs" / "data_sources.yaml")}
    assert sources["cml"].source_type == "skeleton"
    assert sources["cwpv"].source_type == "video_and_imu"


def test_invalid_source_raises():
    with pytest.raises(SourceRegistryError):
        DataSource.from_dict({"id": "bad"})
