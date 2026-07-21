"""Manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worker_activity.data.schema import CLIP_INVENTORY_COLUMNS, normalize_inventory_frame


def write_inventory_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_inventory_frame(df)
    normalized.to_csv(path, index=False)
    return path


def write_inventory_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_inventory_frame(df)
    normalized.to_parquet(path, index=False)
    return path


def read_inventory_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=CLIP_INVENTORY_COLUMNS)
    return pd.read_csv(path)


def manifest_uses_relative_paths(df: pd.DataFrame, repo_root: Path) -> list[str]:
    """Return issues if manifest contains absolute Windows/Unix paths."""
    issues: list[str] = []
    if "relative_path" not in df.columns or df.empty:
        return issues
    for value in df["relative_path"].dropna().astype(str):
        if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
            issues.append(f"Absolute path in manifest: {value}")
        try:
            if Path(value).is_absolute():
                issues.append(f"Absolute path in manifest: {value}")
        except OSError:
            pass
    return issues
