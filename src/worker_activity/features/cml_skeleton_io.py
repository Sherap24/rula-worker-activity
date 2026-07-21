"""CML skeleton JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SkeletonLoadError(Exception):
    """Raised when a CML skeleton file cannot be loaded."""


def load_cml_skeleton(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SkeletonLoadError(str(exc)) from exc
    if not isinstance(data, dict) or "bdata" not in data:
        raise SkeletonLoadError(f"Missing bdata in {path}")
    return data


def bone_positions(bdata: dict, bone: str, frame_idx: int) -> np.ndarray:
    """Return (3,) array x,y,z for bone at frame."""
    entry = bdata[bone]
    return np.array(
        [entry["x"][frame_idx], entry["y"][frame_idx], entry["z"][frame_idx]],
        dtype=np.float64,
    )


def bone_series(bdata: dict, bone: str) -> np.ndarray:
    """Return (T, 3) array for bone across all frames."""
    entry = bdata[bone]
    return np.column_stack([entry["x"], entry["y"], entry["z"]]).astype(np.float64)


def resolve_data_path(relative_path: str, data_root: Path) -> Path:
    return (
        data_root / relative_path.replace("/", "\\")
        if "\\" in str(data_root)
        else data_root / relative_path
    )
