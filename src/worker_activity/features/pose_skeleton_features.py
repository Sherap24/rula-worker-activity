"""Kinematic feature extraction from MediaPipe pose landmark sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _landmark_point(landmarks: dict[str, dict[str, float]], name: str) -> np.ndarray | None:
    lm = landmarks.get(name)
    if lm is None:
        return None
    x, y, z = lm.get("x"), lm.get("y"), lm.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


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


def hip_center(landmarks: dict[str, dict[str, float]]) -> np.ndarray | None:
    left = _landmark_point(landmarks, "LEFT_HIP")
    right = _landmark_point(landmarks, "RIGHT_HIP")
    if left is None or right is None:
        return None
    return (left + right) / 2.0


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


def _joint_angle_series(
    frames: list[dict[str, dict[str, float]]],
    a_name: str,
    b_name: str,
    c_name: str,
) -> np.ndarray:
    values = np.empty(len(frames), dtype=np.float64)
    for idx, landmarks in enumerate(frames):
        a = _landmark_point(landmarks, a_name)
        b = _landmark_point(landmarks, b_name)
        c = _landmark_point(landmarks, c_name)
        if a is None or b is None or c is None:
            values[idx] = np.nan
        else:
            values[idx] = angle_at_joint_deg(a, b, c)
    return values


def pose_frames_from_parquet(df: pd.DataFrame) -> list[dict[str, dict[str, float]]]:
    """Convert long-format pose parquet rows to per-frame landmark dicts."""
    if df.empty:
        return []
    frames: list[dict[str, dict[str, float]]] = []
    for frame_index in sorted(df["frame_index"].unique()):
        frame_rows = df[df["frame_index"] == frame_index]
        landmarks: dict[str, dict[str, float]] = {}
        for _, row in frame_rows.iterrows():
            landmarks[str(row["landmark"])] = {
                "x": float(row["x"]) if pd.notna(row.get("x")) else None,
                "y": float(row["y"]) if pd.notna(row.get("y")) else None,
                "z": float(row["z"]) if pd.notna(row.get("z")) else None,
                "visibility": float(row["visibility"]) if pd.notna(row.get("visibility")) else None,
            }
        frames.append(landmarks)
    return frames


def extract_pose_sequence_features(frames: list[dict[str, dict[str, float]]]) -> dict[str, float]:
    """Extract label-agnostic kinematic features from one pose sequence."""
    features: dict[str, float] = {"frame_count_extracted": float(len(frames))}
    if not frames:
        return features

    visibility_values = []
    for landmarks in frames:
        for lm in landmarks.values():
            vis = lm.get("visibility")
            if vis is not None:
                visibility_values.append(float(vis))
    if visibility_values:
        features.update(_series_stats(np.array(visibility_values), "landmark_visibility"))

    angle_defs = [
        ("left_knee_flexion_deg", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        ("right_knee_flexion_deg", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        ("left_elbow_flexion_deg", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        ("right_elbow_flexion_deg", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        ("left_shoulder_flexion_deg", "LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
        ("right_shoulder_flexion_deg", "RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ]
    for name, a_name, b_name, c_name in angle_defs:
        values = _joint_angle_series(frames, a_name, b_name, c_name)
        features.update(_series_stats(values, name))

    hip_positions = []
    head_heights = []
    for landmarks in frames:
        hc = hip_center(landmarks)
        if hc is not None:
            hip_positions.append(hc)
        nose = _landmark_point(landmarks, "NOSE")
        if nose is not None:
            head_heights.append(nose[1])

    if hip_positions:
        hip_arr = np.array(hip_positions)
        displacements = np.linalg.norm(np.diff(hip_arr, axis=0), axis=1)
        features.update(_series_stats(displacements, "hip_center_speed_per_frame"))
        features.update(_series_stats(hip_arr[:, 1], "hip_center_y"))
        features.update(_series_stats(hip_arr[:, 2], "hip_center_z"))

    if head_heights:
        features.update(_series_stats(np.array(head_heights), "head_height_y"))

    left_knee = features.get("left_knee_flexion_deg_mean", float("nan"))
    right_knee = features.get("right_knee_flexion_deg_mean", float("nan"))
    if not np.isnan(left_knee) and not np.isnan(right_knee):
        features["knee_flexion_asymmetry_deg"] = abs(left_knee - right_knee)
    else:
        features["knee_flexion_asymmetry_deg"] = float("nan")

    return features


def sliding_window_features(
    frames: list[dict[str, dict[str, float]]],
    *,
    window_size: int,
    stride: int,
) -> list[dict[str, float]]:
    """Extract per-window feature vectors from a pose sequence."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(frames) < window_size:
        return []

    windows: list[dict[str, float]] = []
    for start in range(0, len(frames) - window_size + 1, stride):
        chunk = frames[start : start + window_size]
        feats = extract_pose_sequence_features(chunk)
        feats["window_start_frame"] = float(start)
        feats["window_end_frame"] = float(start + window_size - 1)
        windows.append(feats)
    return windows


def aggregate_window_features(windows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate sliding-window features to one video-level vector (mean across windows)."""
    if not windows:
        return {"window_count": 0.0}
    skip = {"window_start_frame", "window_end_frame"}
    keys = [k for k in windows[0] if k not in skip]
    out: dict[str, float] = {"window_count": float(len(windows))}
    for key in keys:
        values = np.array([w[key] for w in windows if key in w], dtype=np.float64)
        clean = values[~np.isnan(values)]
        out[f"{key}_window_mean"] = float(np.mean(clean)) if clean.size else float("nan")
        out[f"{key}_window_std"] = float(np.std(clean)) if clean.size else float("nan")
    return out


def extract_video_pose_features(
    pose_df: pd.DataFrame,
    *,
    window_size: int = 30,
    stride: int = 15,
) -> dict[str, Any]:
    """Full video feature extraction: sequence + sliding-window aggregate."""
    frames = pose_frames_from_parquet(pose_df)
    seq_feats = extract_pose_sequence_features(frames)
    windows = sliding_window_features(frames, window_size=window_size, stride=stride)
    win_agg = aggregate_window_features(windows)
    return {**seq_feats, **win_agg, "sliding_window_count": float(len(windows))}
