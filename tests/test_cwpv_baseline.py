"""Tests for CWPV baseline splits and pose feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worker_activity.data.cml_baseline import audit_leakage
from worker_activity.data.cwpv_baseline import (
    build_cwpv_splits,
    enrich_cwpv_inventory_rows,
)
from worker_activity.data.cwpv_label_map import load_cwpv_label_map
from worker_activity.features.pose_skeleton_features import (
    extract_pose_sequence_features,
    extract_video_pose_features,
    pose_frames_from_parquet,
    sliding_window_features,
)
from worker_activity.models.baseline_classifier import (
    _feature_columns,
    train_cwpv_baseline_classifiers,
)


def _synthetic_cwpv_inventory() -> pd.DataFrame:
    rows = []
    subjects = ["01", "02", "03", "04", "05", "06"]
    motions = ["1", "2", "3", "4"]
    for subject in subjects:
        for motion in motions:
            for trial in ["1", "2"]:
                for camera in ["1", "2"]:
                    # 5-digit extracted form: PP + M + block + T (block fixed at 1)
                    fname = f"{subject}{motion}1{trial}Camera_{camera}.avi"
                    video_id = f"P{subject}_M{motion}_B1_T{trial}"
                    rows.append(
                        {
                            "source": "cwpv",
                            "source_type": "video",
                            "relative_path": f"cwpv/extracted/Video Data/{fname}",
                            "file_name": fname,
                            "extension": ".avi",
                            "video_id": video_id,
                            "clip_id": fname.replace(".avi", ""),
                            "view_id": f"camera_{camera}",
                            "raw_activity_label": f"M{motion}_test",
                            "integrity_status": "ok",
                            "metadata_status": "extracted",
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def cwpv_label_map(repo_root):
    return load_cwpv_label_map(repo_root / "configs" / "label_map_cwpv.yaml")


def test_enrich_cwpv_sets_logical_sample_id(cwpv_label_map):
    df = _synthetic_cwpv_inventory()
    enriched = enrich_cwpv_inventory_rows(df, cwpv_label_map)
    sample = enriched.iloc[0]
    assert sample["logical_sample_id"] == sample["video_id"]
    assert sample["subject_id"] == "01"
    assert sample["motion_id"] in {"1", "2", "3", "4"}


def test_cwpv_subject_disjoint_splits(cwpv_label_map):
    df = _synthetic_cwpv_inventory()
    enriched = enrich_cwpv_inventory_rows(df, cwpv_label_map)
    manifests, meta = build_cwpv_splits(enriched)
    assert meta["unique_subjects_split"] >= 3
    train_subjects = set(manifests["train"]["subject_id"])
    val_subjects = set(manifests["validation"]["subject_id"])
    test_subjects = set(manifests["test"]["subject_id"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)


def test_cwpv_logical_samples_share_split(cwpv_label_map):
    df = _synthetic_cwpv_inventory()
    enriched = enrich_cwpv_inventory_rows(df, cwpv_label_map)
    manifests, _ = build_cwpv_splits(enriched)
    train = manifests["train"]
    for logical_id in train["logical_sample_id"].unique():
        all_views = enriched[enriched["logical_sample_id"] == logical_id]
        train_views = train[train["logical_sample_id"] == logical_id]
        assert len(train_views) == len(all_views[all_views["include_in_baseline"] == True])


def test_cwpv_leakage_audit_passes(cwpv_label_map):
    df = _synthetic_cwpv_inventory()
    enriched = enrich_cwpv_inventory_rows(df, cwpv_label_map)
    manifests, _ = build_cwpv_splits(enriched)
    _, violations = audit_leakage(manifests)
    assert violations == []


def _synthetic_pose_parquet(n_frames: int = 60) -> pd.DataFrame:
    records = []
    for frame_idx in range(n_frames):
        for name in ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "NOSE"]:
            records.append(
                {
                    "frame_index": frame_idx,
                    "landmark": name,
                    "x": 0.5,
                    "y": 0.5 + 0.01 * frame_idx,
                    "z": 0.0,
                    "visibility": 0.9,
                }
            )
    return pd.DataFrame(records)


def test_pose_sequence_features():
    frames = pose_frames_from_parquet(_synthetic_pose_parquet(30))
    feats = extract_pose_sequence_features(frames)
    assert feats["frame_count_extracted"] == 30.0
    assert "left_knee_flexion_deg_mean" in feats


def test_sliding_window_features():
    frames = pose_frames_from_parquet(_synthetic_pose_parquet(60))
    windows = sliding_window_features(frames, window_size=10, stride=5)
    assert len(windows) > 0
    video_feats = extract_video_pose_features(_synthetic_pose_parquet(60), window_size=10, stride=5)
    assert video_feats["sliding_window_count"] > 0


def test_baseline_classifier_train_val_only(tmp_path, repo_root):
    features = []
    for split in ["train", "validation"]:
        for idx in range(6):
            row = {
                "split": split,
                "canonical_activity": ["carrying", "squatting", "kneeling"][idx % 3],
                "relative_path": f"video_{split}_{idx}.avi",
                "feat_a": float(idx),
                "feat_b": float(idx) * 2.0,
            }
            features.append(row)
    feat_df = pd.DataFrame(features)
    feat_path = tmp_path / "features.parquet"
    feat_df.to_parquet(feat_path, index=False)

    result = train_cwpv_baseline_classifiers(features_path=feat_path, persist_models=False)
    assert "logistic_regression" in result.metrics
    assert "random_forest" in result.metrics
    assert result.report_path.is_file()
    assert any("held out" in w.lower() for w in result.warnings) or result.test_manifest_path is None


def test_feature_columns_excludes_metadata():
    df = pd.DataFrame(
        {
            "canonical_activity": ["carrying"],
            "relative_path": ["a.avi"],
            "feat_a": [1.0],
        }
    )
    cols = _feature_columns(df)
    assert cols == ["feat_a"]
