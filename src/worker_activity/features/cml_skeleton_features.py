"""Layout-aware kinematic feature extraction from CML 3D skeleton JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from worker_activity.features.cml_skeleton_io import bone_positions, bone_series


@dataclass(frozen=True)
class LayoutConfig:
    skeleton_layout: int
    bones: list[str]
    hip_center_bones: list[str]
    angles: list[dict[str, Any]]


def angle_at_joint_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by segments ba and bc."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def hip_center_position(bdata: dict, frame_idx: int, layout: LayoutConfig) -> np.ndarray:
    if layout.skeleton_layout == 15:
        return (bone_positions(bdata, "Lefthip", frame_idx) + bone_positions(bdata, "Righthip", frame_idx)) / 2.0
    return bone_positions(bdata, "Hipcentre", frame_idx)


def hip_center_series(bdata: dict, layout: LayoutConfig) -> np.ndarray:
    if layout.skeleton_layout == 15:
        return (bone_series(bdata, "Lefthip") + bone_series(bdata, "Righthip")) / 2.0
    return bone_series(bdata, "Hipcentre")


def resolve_angle_points(
    bdata: dict, frame_idx: int, points: list[str], layout: LayoutConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved: list[np.ndarray] = []
    for name in points:
        if name == "Hipcentre_proxy":
            resolved.append(hip_center_position(bdata, frame_idx, layout))
        else:
            resolved.append(bone_positions(bdata, name, frame_idx))
    return resolved[0], resolved[1], resolved[2]


def _series_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_range": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": float(np.std(clean)),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
        f"{prefix}_range": float(np.max(clean) - np.min(clean)),
    }


def extract_sequence_features(bdata: dict, layout: LayoutConfig) -> dict[str, float]:
    """Extract label-agnostic kinematic features from one skeleton sequence."""
    n_frames = len(bdata[layout.bones[0]]["x"])
    features: dict[str, float] = {"frame_count_extracted": float(n_frames)}

    # Joint angles over time
    for angle_def in layout.angles:
        name = angle_def["name"]
        values = np.empty(n_frames, dtype=np.float64)
        for t in range(n_frames):
            try:
                a, b, c = resolve_angle_points(bdata, t, angle_def["points"], layout)
                values[t] = angle_at_joint_deg(a, b, c)
            except (KeyError, IndexError):
                values[t] = np.nan
        features.update(_series_stats(values, name))

    # Hip center kinematics
    hip = hip_center_series(bdata, layout)
    displacements = np.linalg.norm(np.diff(hip, axis=0), axis=1)
    features.update(_series_stats(displacements, "hip_center_speed_per_frame"))
    features.update(_series_stats(hip[:, 2], "hip_center_height"))
    features.update(_series_stats(hip[:, 0], "hip_center_x"))
    features.update(_series_stats(hip[:, 1], "hip_center_y"))

    # Head height
    head = bone_series(bdata, "Head")
    features.update(_series_stats(head[:, 2], "head_height"))

    # Symmetry proxy: left vs right knee angle mean difference
    left_knee = features.get("left_knee_flexion_deg_mean", float("nan"))
    right_knee = features.get("right_knee_flexion_deg_mean", float("nan"))
    if not np.isnan(left_knee) and not np.isnan(right_knee):
        features["knee_flexion_asymmetry_deg"] = abs(left_knee - right_knee)
    else:
        features["knee_flexion_asymmetry_deg"] = float("nan")

    return features


def layout_config_from_yaml(entry: dict[str, Any]) -> LayoutConfig:
    return LayoutConfig(
        skeleton_layout=int(entry["skeleton_layout"]),
        bones=list(entry["bones"]),
        hip_center_bones=list(entry["hip_center_bones"]),
        angles=list(entry["angles"]),
    )
