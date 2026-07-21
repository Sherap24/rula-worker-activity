"""Tests for smartphone metadata and domain-transfer evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worker_activity.data.smartphone_metadata import (
    enrich_smartphone_row,
    parse_smartphone_filename,
)
from worker_activity.models.baseline_classifier import (
    save_cwpv_baseline_artifacts,
)
from worker_activity.models.domain_transfer import evaluate_domain_transfer


def test_parse_smartphone_filename():
    parsed = parse_smartphone_filename("squatting_01.mp4")
    assert parsed is not None
    assert parsed.canonical_activity == "squatting"
    assert parsed.clip_id == "01"


def test_parse_smartphone_overhead():
    parsed = parse_smartphone_filename("overhead_work_reaching_02.MP4")
    assert parsed is not None
    assert parsed.canonical_activity == "overhead_work_reaching"


def test_parse_smartphone_invalid():
    assert parse_smartphone_filename("random.mp4") is None
    assert parse_smartphone_filename("walking_01.mp4") is None


def test_enrich_smartphone_row(tmp_path: Path):
    video = tmp_path / "carrying_01.mp4"
    video.write_bytes(b"fake")
    row = enrich_smartphone_row({"notes": None}, video)
    assert row["canonical_activity"] == "carrying"
    assert row["subject_id"] == "phone_self"
    assert row["include_in_baseline"] is False
    assert row["label_mapping_status"] == "mapped"


def test_domain_transfer_eval_synthetic(tmp_path: Path, monkeypatch, repo_root: Path):
    # Minimal 2-class features matching train/eval path
    rng = np.random.default_rng(0)
    feature_cols = [f"f{i}" for i in range(4)]
    labels = ["carrying", "squatting"]

    train_rows = []
    for label in labels:
        for _ in range(20):
            vals = rng.normal(0 if label == "carrying" else 3, 0.3, size=4)
            train_rows.append({**dict(zip(feature_cols, vals)), "canonical_activity": label})
    train_df = pd.DataFrame(train_rows)

    models = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "random_forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=20, random_state=0))]
        ),
    }
    x = train_df[feature_cols].to_numpy()
    y = train_df["canonical_activity"].to_numpy()
    for model in models.values():
        model.fit(x, y)

    model_dir = tmp_path / "models"
    save_cwpv_baseline_artifacts(
        models=models,
        feature_cols=feature_cols,
        label_column="canonical_activity",
        metrics={
            "logistic_regression": {"accuracy": 1.0, "macro_f1": 1.0},
            "random_forest": {"accuracy": 1.0, "macro_f1": 1.0},
        },
        model_dir=model_dir,
    )

    phone_rows = []
    for label in labels:
        for i in range(2):
            vals = rng.normal(0 if label == "carrying" else 3, 0.3, size=4)
            phone_rows.append(
                {
                    **dict(zip(feature_cols, vals)),
                    "canonical_activity": label,
                    "relative_path": f"local_smartphone/{label}_{i:02d}.mp4",
                    "file_name": f"{label}_{i:02d}.mp4",
                    "subject_id": "phone_self",
                }
            )
    phone_path = tmp_path / "phone_features.parquet"
    pd.DataFrame(phone_rows).to_parquet(phone_path, index=False)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    class FakePaths:
        pass

    fake = FakePaths()
    fake.reports_dir = reports_dir
    fake.processed_dir = tmp_path
    fake.manifests_dir = tmp_path
    fake.outputs_dir = tmp_path
    fake.data_root = tmp_path

    monkeypatch.setattr(
        "worker_activity.models.domain_transfer.resolve_paths_config",
        lambda **kwargs: fake,
    )
    monkeypatch.setattr(
        "worker_activity.models.domain_transfer.find_repo_root",
        lambda: repo_root,
    )

    result = evaluate_domain_transfer(features_path=phone_path, model_dir=model_dir)
    assert result.report_path.is_file()
    assert result.predictions_path.is_file()
    assert "logistic_regression" in result.metrics
    assert result.metrics["logistic_regression"]["accuracy"] >= 0.5
