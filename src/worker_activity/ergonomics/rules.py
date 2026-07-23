"""Frame-level posture rule helpers for screening indicators."""

from __future__ import annotations

from typing import Any

import numpy as np

from worker_activity.features.pose_skeleton_features import (
    _landmark_point,
    angle_at_joint_deg,
    hip_center,
)


def shoulder_midpoint(landmarks: dict[str, dict[str, float]]) -> np.ndarray | None:
    left = _landmark_point(landmarks, "LEFT_SHOULDER")
    right = _landmark_point(landmarks, "RIGHT_SHOULDER")
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def trunk_flexion_from_vertical_deg(landmarks: dict[str, dict[str, float]]) -> float:
    """Angle between hip→shoulder vector and image vertical (up = -y)."""
    hip = hip_center(landmarks)
    shoulder = shoulder_midpoint(landmarks)
    if hip is None or shoulder is None:
        return float("nan")
    torso = shoulder[:2] - hip[:2]
    n = np.linalg.norm(torso)
    if n < 1e-8:
        return float("nan")
    up = np.array([0.0, -1.0], dtype=np.float64)
    cos_a = float(np.clip(np.dot(torso / n, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def mean_knee_flexion_deg(landmarks: dict[str, dict[str, float]]) -> float:
    left_a = _landmark_point(landmarks, "LEFT_HIP")
    left_b = _landmark_point(landmarks, "LEFT_KNEE")
    left_c = _landmark_point(landmarks, "LEFT_ANKLE")
    right_a = _landmark_point(landmarks, "RIGHT_HIP")
    right_b = _landmark_point(landmarks, "RIGHT_KNEE")
    right_c = _landmark_point(landmarks, "RIGHT_ANKLE")
    vals: list[float] = []
    if left_a is not None and left_b is not None and left_c is not None:
        vals.append(angle_at_joint_deg(left_a, left_b, left_c))
    if right_a is not None and right_b is not None and right_c is not None:
        vals.append(angle_at_joint_deg(right_a, right_b, right_c))
    if not vals:
        return float("nan")
    return float(np.nanmean(np.array(vals, dtype=np.float64)))


def wrist_above_nose(landmarks: dict[str, dict[str, float]], *, margin: float) -> bool:
    nose = _landmark_point(landmarks, "NOSE")
    if nose is None:
        return False
    for name in ("LEFT_WRIST", "RIGHT_WRIST"):
        wrist = _landmark_point(landmarks, name)
        if wrist is not None and (nose[1] - wrist[1]) >= margin:
            return True
    return False


def frame_indicators(
    landmarks: dict[str, dict[str, float]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Boolean posture flags + continuous proxies for one frame."""
    trunk = trunk_flexion_from_vertical_deg(landmarks)
    knee = mean_knee_flexion_deg(landmarks)
    hip = hip_center(landmarks)
    hip_y = float(hip[1]) if hip is not None else float("nan")

    bend_cfg = cfg.get("bending", {})
    overhead_cfg = cfg.get("overhead", {})
    kneel_cfg = cfg.get("kneeling", {})
    squat_cfg = cfg.get("squatting", {})
    awkward_cfg = cfg.get("awkward", {})

    bending = (not np.isnan(trunk)) and trunk >= float(bend_cfg.get("trunk_flexion_deg_min", 35.0))
    overhead = wrist_above_nose(
        landmarks,
        margin=float(overhead_cfg.get("wrist_above_nose_margin", 0.02)),
    )
    kneeling = (
        (not np.isnan(knee))
        and knee <= float(kneel_cfg.get("knee_flexion_deg_max", 120.0))
        and (not np.isnan(hip_y))
        and hip_y >= float(kneel_cfg.get("hip_y_min", 0.45))
        and not overhead
    )
    squatting = (
        (not np.isnan(knee))
        and knee <= float(squat_cfg.get("knee_flexion_deg_max", 110.0))
        and (not np.isnan(hip_y))
        and hip_y >= float(squat_cfg.get("hip_y_min", 0.50))
        and not overhead
    )
    awkward = (
        ((not np.isnan(trunk)) and trunk >= float(awkward_cfg.get("trunk_flexion_deg_min", 45.0)))
        or ((not np.isnan(knee)) and knee <= float(awkward_cfg.get("knee_flexion_deg_max", 100.0)))
    )

    return {
        "trunk_flexion_deg": trunk,
        "knee_flexion_deg": knee,
        "hip_y": hip_y,
        "bending": bending,
        "overhead": overhead,
        "kneeling": kneeling,
        "squatting": squatting,
        "awkward": awkward,
    }


def count_bouts(flags: list[bool], *, min_consecutive: int) -> int:
    """Count rising-edge bouts lasting at least min_consecutive frames."""
    if min_consecutive < 1:
        min_consecutive = 1
    count = 0
    run = 0
    counted = False
    for flag in flags:
        if flag:
            run += 1
            if run >= min_consecutive and not counted:
                count += 1
                counted = True
        else:
            run = 0
            counted = False
    return count


def duration_seconds(flags: list[bool], *, fps: float, min_consecutive: int) -> float:
    """Sum durations of runs lasting at least min_consecutive frames."""
    if fps <= 0:
        fps = 30.0
    if min_consecutive < 1:
        min_consecutive = 1
    total_frames = 0
    run = 0
    for flag in flags:
        if flag:
            run += 1
        else:
            if run >= min_consecutive:
                total_frames += run
            run = 0
    if run >= min_consecutive:
        total_frames += run
    return float(total_frames) / float(fps)
