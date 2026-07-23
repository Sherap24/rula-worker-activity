"""Unit tests for Week 6 ergonomic screening and zone events."""

from __future__ import annotations

import numpy as np

from worker_activity.ergonomics.rules import (
    count_bouts,
    duration_seconds,
    frame_indicators,
    trunk_flexion_from_vertical_deg,
)
from worker_activity.ergonomics.screening import screen_pose_frames
from worker_activity.zones.events import detect_events_for_frames
from worker_activity.zones.geometry import point_in_polygon


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _upright_landmarks() -> dict[str, dict[str, float]]:
    """Standing pose, hips mid-frame, shoulders above hips."""
    return {
        "NOSE": _lm(0.5, 0.20),
        "LEFT_SHOULDER": _lm(0.40, 0.35),
        "RIGHT_SHOULDER": _lm(0.60, 0.35),
        "LEFT_ELBOW": _lm(0.35, 0.50),
        "RIGHT_ELBOW": _lm(0.65, 0.50),
        "LEFT_WRIST": _lm(0.35, 0.60),
        "RIGHT_WRIST": _lm(0.65, 0.60),
        "LEFT_HIP": _lm(0.42, 0.55),
        "RIGHT_HIP": _lm(0.58, 0.55),
        "LEFT_KNEE": _lm(0.42, 0.75),
        "RIGHT_KNEE": _lm(0.58, 0.75),
        "LEFT_ANKLE": _lm(0.42, 0.95),
        "RIGHT_ANKLE": _lm(0.58, 0.95),
    }


def _bent_landmarks() -> dict[str, dict[str, float]]:
    """Forward lean: shoulders shifted down/forward relative to hips."""
    lm = _upright_landmarks()
    lm["LEFT_SHOULDER"] = _lm(0.40, 0.58)
    lm["RIGHT_SHOULDER"] = _lm(0.60, 0.58)
    lm["NOSE"] = _lm(0.50, 0.50)
    return lm


def _overhead_landmarks() -> dict[str, dict[str, float]]:
    lm = _upright_landmarks()
    lm["LEFT_WRIST"] = _lm(0.40, 0.10)
    lm["RIGHT_WRIST"] = _lm(0.60, 0.10)
    return lm


def _squat_landmarks() -> dict[str, dict[str, float]]:
    lm = _upright_landmarks()
    lm["LEFT_HIP"] = _lm(0.42, 0.70)
    lm["RIGHT_HIP"] = _lm(0.58, 0.70)
    lm["LEFT_KNEE"] = _lm(0.35, 0.80)
    lm["RIGHT_KNEE"] = _lm(0.65, 0.80)
    lm["LEFT_ANKLE"] = _lm(0.42, 0.95)
    lm["RIGHT_ANKLE"] = _lm(0.58, 0.95)
    lm["LEFT_SHOULDER"] = _lm(0.40, 0.50)
    lm["RIGHT_SHOULDER"] = _lm(0.60, 0.50)
    lm["NOSE"] = _lm(0.50, 0.40)
    return lm


DEFAULT_CFG = {
    "min_consecutive_frames": 2,
    "default_fps": 10.0,
    "bending": {"trunk_flexion_deg_min": 35.0},
    "overhead": {"wrist_above_nose_margin": 0.02},
    "kneeling": {"knee_flexion_deg_max": 120.0, "hip_y_min": 0.45},
    "squatting": {"knee_flexion_deg_max": 110.0, "hip_y_min": 0.50},
    "awkward": {"trunk_flexion_deg_min": 45.0, "knee_flexion_deg_max": 100.0},
}


def test_point_in_polygon_square():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon(0.5, 0.5, square) is True
    assert point_in_polygon(1.5, 0.5, square) is False


def test_trunk_flexion_increases_when_bent():
    upright = trunk_flexion_from_vertical_deg(_upright_landmarks())
    bent = trunk_flexion_from_vertical_deg(_bent_landmarks())
    assert upright < 20.0
    assert bent > upright


def test_frame_indicators_overhead_and_bend():
    overhead = frame_indicators(_overhead_landmarks(), DEFAULT_CFG)
    bent = frame_indicators(_bent_landmarks(), DEFAULT_CFG)
    upright = frame_indicators(_upright_landmarks(), DEFAULT_CFG)
    assert overhead["overhead"] is True
    assert upright["overhead"] is False
    assert bent["bending"] is True
    assert upright["bending"] is False


def test_count_bouts_and_duration():
    flags = [False, True, True, True, False, True, True, False]
    assert count_bouts(flags, min_consecutive=2) == 2
    assert count_bouts(flags, min_consecutive=3) == 1
    assert duration_seconds([True, True, True, False], fps=10.0, min_consecutive=2) == 0.3


def test_screen_pose_frames_bending_bouts():
    frames = (
        [_upright_landmarks()] * 2
        + [_bent_landmarks()] * 3
        + [_upright_landmarks()] * 2
        + [_bent_landmarks()] * 3
    )
    metrics = screen_pose_frames(frames, DEFAULT_CFG, fps=10.0)
    assert metrics["repeated_bending_count"] == 2.0
    assert metrics["bending_duration_s"] > 0.0


def test_detect_zone_entry_exit():
    # Body center for upright landmarks is ~ (0.5, 0.55)
    poly = {
        "id": "demo",
        "label": "demo",
        "vertices": [(0.4, 0.4), (0.9, 0.4), (0.9, 0.9), (0.4, 0.9)],
    }
    outside = _upright_landmarks()
    outside["LEFT_HIP"] = _lm(0.10, 0.55)
    outside["RIGHT_HIP"] = _lm(0.20, 0.55)
    inside = _upright_landmarks()
    inside["LEFT_HIP"] = _lm(0.50, 0.55)
    inside["RIGHT_HIP"] = _lm(0.60, 0.55)

    frames = [outside, outside, inside, inside, inside, outside, outside]
    events = detect_events_for_frames(
        frames,
        [poly],
        point_mode="body_center",
        fps=10.0,
        min_consecutive=2,
    )
    types = [e["event_type"] for e in events]
    assert "restricted_zone_entry" in types
    assert "restricted_zone_exit" in types


def test_squat_indicator_fires():
    ind = frame_indicators(_squat_landmarks(), DEFAULT_CFG)
    assert ind["squatting"] is True or ind["kneeling"] is True or ind["awkward"] is True
    assert not np.isnan(ind["knee_flexion_deg"])
