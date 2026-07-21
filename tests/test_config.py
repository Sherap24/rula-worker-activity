"""Tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from worker_activity.config import (
    DATA_ROOT_ENV_VAR,
    ConfigurationError,
    resolve_paths_config,
    source_data_path,
    validate_paths_config,
)


def test_resolve_paths_without_data_root(repo_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(repo_root)
    paths = resolve_paths_config(repo_root=repo_root)
    assert paths.data_root is None
    warnings = validate_paths_config(paths)
    assert any(DATA_ROOT_ENV_VAR in w for w in warnings)


def test_data_root_env_override(repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "external_data"
    data_root.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(data_root))
    monkeypatch.chdir(repo_root)
    paths = resolve_paths_config(repo_root=repo_root)
    assert paths.data_root == data_root.resolve()


def test_source_data_path_relative(repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "RULA"
    data_root.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(data_root))
    paths = resolve_paths_config(repo_root=repo_root)
    cwpv_path = source_data_path(paths, "cwpv", "extracted")
    assert cwpv_path == (data_root / "cwpv" / "extracted").resolve()


def test_missing_repo_root_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError):
        resolve_paths_config(repo_root=tmp_path)
