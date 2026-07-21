"""Configuration loading with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATA_ROOT_ENV_VAR = "RULA_DATA_ROOT"
OUTPUT_DIR_ENV_VAR = "RULA_OUTPUT_DIR"
USE_FFPROBE_ENV_VAR = "RULA_USE_FFPROBE"


class ConfigurationError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass
class PathsConfig:
    repo_root: Path
    data_root: Path | None
    processed_dir: Path
    manifests_dir: Path
    reports_dir: Path
    outputs_dir: Path
    source_dirs: dict[str, dict[str, str]] = field(default_factory=dict)
    metadata_provider: str = "opencv"
    ffprobe_binary: str = "ffprobe"
    inventory_filenames: dict[str, str] = field(default_factory=dict)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* to locate the repository root (contains pyproject.toml)."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            _load_dotenv_file(candidate / ".env")
            return candidate
    raise ConfigurationError(
        "Could not locate repository root (pyproject.toml not found). "
        "Run commands from the project directory."
    )


def _load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding existing env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file must be a mapping: {path}")
    return data


def resolve_paths_config(
    repo_root: Path | None = None,
    paths_file: Path | None = None,
    *,
    require_data_root: bool = False,
) -> PathsConfig:
    root = repo_root or find_repo_root()
    paths_path = paths_file or root / "configs" / "paths.yaml"
    if not paths_path.is_file():
        paths_path = root / "configs" / "paths.example.yaml"
    if not paths_path.is_file():
        raise ConfigurationError(
            "No paths configuration found. Copy configs/paths.example.yaml to "
            "configs/paths.yaml or set environment variables."
        )

    raw = load_yaml(paths_path)
    env_var = raw.get("data_root_env_var", DATA_ROOT_ENV_VAR)
    data_root_value = os.environ.get(env_var) or os.environ.get(DATA_ROOT_ENV_VAR)
    data_root = Path(data_root_value).expanduser().resolve() if data_root_value else None

    if require_data_root and data_root is None:
        raise ConfigurationError(
            f"{DATA_ROOT_ENV_VAR} is not set. "
            "Large datasets must live outside OneDrive. "
            "See docs/DATA_ACQUISITION.md."
        )

    processed = root / raw.get("processed_subdir", "data/processed")
    manifests = root / raw.get("manifests_subdir", "data/manifests")
    reports = root / raw.get("reports_subdir", "reports")
    output_name = os.environ.get(OUTPUT_DIR_ENV_VAR) or raw.get("outputs_subdir", "outputs")
    outputs = root / output_name

    metadata = raw.get("metadata", {})
    provider = metadata.get("provider", "opencv")
    if os.environ.get(USE_FFPROBE_ENV_VAR, "").lower() in {"1", "true", "yes"}:
        provider = "ffprobe"

    inventory = raw.get("inventory", {})

    return PathsConfig(
        repo_root=root,
        data_root=data_root,
        processed_dir=processed,
        manifests_dir=manifests,
        reports_dir=reports,
        outputs_dir=outputs,
        source_dirs=raw.get("source_dirs", {}),
        metadata_provider=provider,
        ffprobe_binary=metadata.get("ffprobe_binary", "ffprobe"),
        inventory_filenames={
            "csv": inventory.get("csv", "clip_inventory.csv"),
            "parquet": inventory.get("parquet", "clip_inventory.parquet"),
            "markdown": inventory.get("markdown", "dataset_inventory.md"),
        },
    )


def source_data_path(paths: PathsConfig, source_id: str, subkey: str = "extracted") -> Path | None:
    """Resolve absolute path for a source subdirectory under RULA_DATA_ROOT."""
    if paths.data_root is None:
        return None
    source_entry = paths.source_dirs.get(source_id, source_id)
    if isinstance(source_entry, str):
        relative = source_entry
    else:
        relative = source_entry.get(subkey) or source_entry.get("extracted") or source_id
    return paths.data_root / relative


def ensure_repo_output_dirs(paths: PathsConfig) -> None:
    """Create repository-local output directories."""
    for directory in (
        paths.processed_dir,
        paths.manifests_dir,
        paths.reports_dir,
        paths.outputs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def validate_paths_config(paths: PathsConfig) -> list[str]:
    """Return list of warnings (empty if fully OK for foundation stage)."""
    warnings: list[str] = []
    if paths.data_root is None:
        warnings.append(
            f"{DATA_ROOT_ENV_VAR} is not set — external dataset directories cannot be audited."
        )
    elif not paths.data_root.is_dir():
        warnings.append(
            f"{DATA_ROOT_ENV_VAR} points to a path that does not exist yet: {paths.data_root}"
        )
    return warnings
