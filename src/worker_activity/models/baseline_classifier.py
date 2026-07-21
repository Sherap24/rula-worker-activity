"""Baseline activity classifiers for CWPV pose features (train/val only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config

DEFAULT_MODEL_DIR = "outputs/models"
MODEL_META_NAME = "cwpv_baseline_meta.yaml"


@dataclass
class BaselineClassifierResult:
    models: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    report_path: Path
    predictions_path: Path
    model_dir: Path | None = None
    test_manifest_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def load_baseline_config(path: Path | None = None) -> dict[str, Any]:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "cwpv_baseline_classifier.yaml"
    if not cfg_path.is_file():
        return {
            "random_seed": 42,
            "label_column": "canonical_activity",
            "exclude_test_from_training": True,
        }
    return load_yaml(cfg_path)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "relative_path",
        "file_name",
        "logical_sample_id",
        "representation_group_id",
        "canonical_activity",
        "raw_activity_label",
        "subject_id",
        "motion_id",
        "view_id",
        "split",
        "frame_count",
        "fps",
        "extraction_status",
        "pose_parquet",
        "source",
        "domain",
        "clip_id",
        "video_id",
    }
    numeric_cols = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def _prepare_xy(
    df: pd.DataFrame,
    *,
    label_column: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    usable = df[df[label_column].notna()].copy()
    x = usable[feature_cols].astype(float).fillna(0.0).to_numpy()
    y = usable[label_column].astype(str).to_numpy()
    return x, y, usable


def cwpv_model_dir(root: Path | None = None) -> Path:
    base = root or find_repo_root()
    return base / DEFAULT_MODEL_DIR


def save_cwpv_baseline_artifacts(
    *,
    models: dict[str, Pipeline],
    feature_cols: list[str],
    label_column: str,
    metrics: dict[str, dict[str, float]],
    model_dir: Path | None = None,
) -> Path:
    """Persist trained pipelines + metadata for domain-transfer evaluation."""
    out = model_dir or cwpv_model_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, out / f"cwpv_baseline_{name}.joblib")
    meta = {
        "label_column": label_column,
        "feature_columns": feature_cols,
        "model_names": sorted(models.keys()),
        "validation_metrics": metrics,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "training_domain": "cwpv",
        "exclude_test_from_training": True,
    }
    (out / MODEL_META_NAME).write_text(
        yaml.safe_dump(meta, sort_keys=False),
        encoding="utf-8",
    )
    return out


def load_cwpv_baseline_artifacts(
    model_dir: Path | None = None,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    """Load saved CWPV baseline pipelines and metadata."""
    out = model_dir or cwpv_model_dir()
    meta_path = out / MODEL_META_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing model metadata: {meta_path}. Run train-cwpv-baseline first."
        )
    meta = load_yaml(meta_path)
    models: dict[str, Pipeline] = {}
    for name in meta.get("model_names", []):
        path = out / f"cwpv_baseline_{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        models[name] = joblib.load(path)
    if not models:
        raise FileNotFoundError(f"No model artifacts found under {out}")
    return models, meta


def train_cwpv_baseline_classifiers(
    features_path: Path | None = None,
    *,
    persist_models: bool = True,
) -> BaselineClassifierResult:
    """Train baseline classifiers on train split; evaluate on validation only."""
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    cfg = load_baseline_config()

    feat_path = features_path or paths.processed_dir / "cwpv" / "features_train_val.parquet"
    if not feat_path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {feat_path}. Run extract-cwpv-features first."
        )

    df = pd.read_parquet(feat_path)
    label_column = str(cfg.get("label_column", "canonical_activity"))
    seed = int(cfg.get("random_seed", 42))

    if "split" not in df.columns:
        raise ValueError("Feature file missing 'split' column.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    warnings: list[str] = []
    if train_df.empty:
        raise ValueError("Train split is empty in feature file.")
    if val_df.empty:
        warnings.append("Validation split is empty — metrics will be unavailable.")

    feature_cols = _feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    x_train, y_train, train_meta = _prepare_xy(
        train_df, label_column=label_column, feature_cols=feature_cols
    )
    x_val, y_val, val_meta = _prepare_xy(
        val_df, label_column=label_column, feature_cols=feature_cols
    )

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    val_predictions: list[pd.DataFrame] = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        if len(val_df) == 0:
            metrics[name] = {"accuracy": float("nan"), "macro_f1": float("nan")}
            continue
        y_pred = model.predict(x_val)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        }
        pred_df = val_meta[
            [
                c
                for c in ["relative_path", "logical_sample_id", "subject_id", label_column]
                if c in val_meta.columns
            ]
        ].copy()
        pred_df["model"] = name
        pred_df["predicted"] = y_pred
        val_predictions.append(pred_df)

    test_manifest = paths.manifests_dir / "cwpv_test.csv"
    if test_manifest.is_file() and cfg.get("exclude_test_from_training", True):
        warnings.append(
            f"Test manifest exists ({test_manifest.name}) but was NOT used for training "
            "or evaluation (held out by design)."
        )

    model_dir: Path | None = None
    if persist_models:
        model_dir = save_cwpv_baseline_artifacts(
            models=models,
            feature_cols=feature_cols,
            label_column=label_column,
            metrics=metrics,
            model_dir=cwpv_model_dir(root),
        )

    out_dir = paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "cwpv_baseline_classifier.md"
    predictions_path = out_dir / "cwpv_baseline_val_predictions.csv"

    if val_predictions:
        pd.concat(val_predictions, ignore_index=True).to_csv(predictions_path, index=False)
    else:
        pd.DataFrame(columns=["relative_path", "model", "predicted"]).to_csv(
            predictions_path, index=False
        )

    _write_classifier_report(
        report_path,
        metrics=metrics,
        train_df=train_meta,
        val_df=val_meta,
        feature_cols=feature_cols,
        label_column=label_column,
        warnings=warnings,
        models=models,
        x_val=x_val,
        y_val=y_val,
    )

    return BaselineClassifierResult(
        models=models,
        metrics=metrics,
        report_path=report_path,
        predictions_path=predictions_path,
        model_dir=model_dir,
        test_manifest_path=test_manifest if test_manifest.is_file() else None,
        warnings=warnings,
    )


def _write_classifier_report(
    path: Path,
    *,
    metrics: dict[str, dict[str, float]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    label_column: str,
    warnings: list[str],
    models: dict[str, Pipeline],
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    lines = [
        "# CWPV Baseline Classifier Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Policy:** Trained on `train` split only. Evaluated on `validation` only. "
        "`test` split is held out.",
        "",
        f"- Train videos: {len(train_df)}",
        f"- Validation videos: {len(val_df)}",
        f"- Feature dimensions: {len(feature_cols)}",
        f"- Label column: `{label_column}`",
        "",
        "## Validation metrics",
        "",
        "| Model | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")

    if len(val_df) > 0:
        lines.extend(["", "## Classification report (validation)", ""])
        best_name = max(metrics, key=lambda k: metrics[k].get("macro_f1", 0.0))
        best_model = models[best_name]
        y_pred = best_model.predict(x_val)
        report = classification_report(y_val, y_pred, zero_division=0)
        lines.append(f"Best model by macro F1: `{best_name}`")
        lines.append("")
        lines.append("```")
        lines.append(report.rstrip())
        lines.append("```")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")

    path.write_text("\n".join(lines), encoding="utf-8")
