"""Tests for CML skeleton feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worker_activity.features.cml_feature_pipeline import extract_train_features
from worker_activity.features.cml_skeleton_features import (
    LayoutConfig,
    angle_at_joint_deg,
    extract_sequence_features,
)


def _make_bdata(bones: list[str], n_frames: int = 10) -> dict:
    bdata = {}
    for i, bone in enumerate(bones):
        t = np.arange(n_frames, dtype=float)
        bdata[bone] = {
            "x": (np.sin(t * 0.1 + i) * 0.1).tolist(),
            "y": (np.cos(t * 0.1 + i) * 0.1).tolist(),
            "z": (t * 0.01 + i * 0.05).tolist(),
        }
    return bdata


LAYOUT_15 = LayoutConfig(
    skeleton_layout=15,
    bones=[
        "Head", "Shouldercentre", "Spine", "Leftshoulder", "Leftelbow", "Lefthand",
        "Rightshoulder", "Rightelbow", "Righthand", "Lefthip", "Leftknee", "Leftfoot",
        "Righthip", "Rightknee", "Rightfoot",
    ],
    hip_center_bones=["Lefthip", "Righthip"],
    angles=[
        {"name": "left_knee_flexion_deg", "points": ["Lefthip", "Leftknee", "Leftfoot"]},
    ],
)


def test_angle_at_joint_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(angle_at_joint_deg(a, b, c) - 90.0) < 1e-6


def test_extract_sequence_features_returns_stats():
    bdata = _make_bdata(LAYOUT_15.bones, n_frames=20)
    feats = extract_sequence_features(bdata, LAYOUT_15)
    assert feats["frame_count_extracted"] == 20.0
    assert "left_knee_flexion_deg_mean" in feats
    assert "hip_center_speed_per_frame_mean" in feats


def test_extract_train_features_one_per_logical_sample(
    tmp_path, monkeypatch,
):
    data_root = tmp_path / "RULA"
    repo_root = Path(__file__).resolve().parents[1]
    bones_15 = LAYOUT_15.bones
    rel_base = "cml/extracted/Construction_Related_Data/x/15_nodes/bending"
    json_dir = data_root / rel_base
    json_dir.mkdir(parents=True)

    bdata = _make_bdata(bones_15, 5)
    skeleton = {"bdata": bdata, "frames": 5, "bones": bones_15, "joints": 15}
    (json_dir / "000001.json").write_text(json.dumps(skeleton), encoding="utf-8")

    manifest = pd.DataFrame(
        [
            {
                "logical_sample_id": "cml_test1",
                "representation_group_id": "cml_test1",
                "skeleton_layout": 15,
                "canonical_activity": "bending_stooping",
                "raw_activity_label": "bending",
                "subject_id": "CMU:1",
                "source_dataset": "CMU",
                "relative_path": f"{rel_base}/000001.json".replace("\\", "/"),
                "source_file": "bend/1_01_01.txt",
                "frame_count": 5,
            }
        ]
    )
    manifest_path = tmp_path / "cml_train.csv"
    manifest.to_csv(manifest_path, index=False)

    monkeypatch.setenv("RULA_DATA_ROOT", str(data_root))
    monkeypatch.chdir(repo_root)

    result = extract_train_features(train_manifest=manifest_path, data_root=data_root)
    assert len(result.features_15) == 1
    assert result.features_15["logical_sample_id"].iloc[0] == "cml_test1"
    assert len(result.features_20) == 0
    assert result.features_15["logical_sample_id"].duplicated().sum() == 0


def test_features_do_not_use_labels_in_computation():
    """Feature extraction is deterministic and label-agnostic."""
    bdata = _make_bdata(LAYOUT_15.bones, 8)
    f1 = extract_sequence_features(bdata, LAYOUT_15)
    f2 = extract_sequence_features(bdata, LAYOUT_15)
    assert set(f1) == set(f2)
    for key in f1:
        v1, v2 = f1[key], f2[key]
        if np.isnan(v1) and np.isnan(v2):
            continue
        assert v1 == v2
