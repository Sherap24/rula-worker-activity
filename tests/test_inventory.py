"""Tests for inventory building."""

import os

import pytest

from worker_activity.config import DATA_ROOT_ENV_VAR, resolve_paths_config
from worker_activity.data.inventory import audit_data_sources, build_inventory, detect_local_status
from worker_activity.data.manifests import manifest_uses_relative_paths
from worker_activity.data.source_registry import load_source_registry


def test_build_inventory_without_data_root(repo_root, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(repo_root)
    paths = resolve_paths_config(repo_root=repo_root)
    result = build_inventory(paths)
    assert len(result.frame) == 0
    assert any("RULA_DATA_ROOT" in w for w in result.warnings)


def test_build_inventory_empty_external_dirs(repo_root, external_data_root, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(external_data_root))
    monkeypatch.chdir(repo_root)
    paths = resolve_paths_config(repo_root=repo_root)
    result = build_inventory(paths)
    assert len(result.frame) == 0
    assert result.files_scanned == 0


def test_detect_not_downloaded(repo_root, external_data_root, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(external_data_root))
    paths = resolve_paths_config(repo_root=repo_root)
    cml = next(s for s in load_source_registry() if s.id == "cml")
    assert detect_local_status(cml, paths) == "not_downloaded"


def test_audit_data_sources(repo_root, external_data_root, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(external_data_root))
    paths = resolve_paths_config(repo_root=repo_root)
    records = audit_data_sources(paths)
    assert any(r["id"] == "cwpv" for r in records)


def test_manifest_no_absolute_paths(repo_root):
    import pandas as pd

    from worker_activity.data.schema import inventory_row, normalize_inventory_frame

    row = inventory_row(
        source="cwpv",
        source_type="video_and_imu",
        relative_path="cwpv/extracted/subject01/task01.mp4",
        file_name="task01.mp4",
        extension=".mp4",
    )
    df = normalize_inventory_frame(pd.DataFrame([row]))
    issues = manifest_uses_relative_paths(df, repo_root)
    assert issues == []
