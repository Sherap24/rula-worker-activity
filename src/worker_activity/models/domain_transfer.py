"""Evaluate CWPV-trained baselines on smartphone features (Week 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from worker_activity.config import find_repo_root, resolve_paths_config
from worker_activity.models.baseline_classifier import (
    _feature_columns,
    load_cwpv_baseline_artifacts,
)


@dataclass
class DomainTransferResult:
    metrics: dict[str, dict[str, float]]
    report_path: Path
    predictions_path: Path
    warnings: list[str] = field(default_factory=list)


def evaluate_domain_transfer(
    *,
    features_path: Path | None = None,
    model_dir: Path | None = None,
) -> DomainTransferResult:
    """Score smartphone features with CWPV-trained models (eval-only)."""
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root)

    feat_path = features_path or (
        paths.processed_dir / "local_smartphone" / "features.parquet"
    )
    if not feat_path.is_file():
        raise FileNotFoundError(
            f"Smartphone features not found: {feat_path}. "
            "Run extract-smartphone-features first."
        )

    models, meta = load_cwpv_baseline_artifacts(model_dir=model_dir)
    label_column = str(meta.get("label_column", "canonical_activity"))
    feature_cols: list[str] = list(meta.get("feature_columns") or [])
    cwpv_val_metrics: dict[str, dict[str, float]] = meta.get("validation_metrics") or {}

    df = pd.read_parquet(feat_path)
    warnings: list[str] = [
        "CWPV test split was not used for training or domain-transfer evaluation.",
        "Smartphone clips are evaluation-only (no retraining).",
    ]

    if not feature_cols:
        feature_cols = _feature_columns(df)
        warnings.append("Model metadata lacked feature_columns; inferred from phone features.")

    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = 0.0
    if missing:
        warnings.append(
            f"Filled {len(missing)} missing feature columns with 0.0 for alignment."
        )

    usable = df[df[label_column].notna()].copy()
    if usable.empty:
        raise ValueError(f"No labeled rows in {feat_path} (column `{label_column}`).")

    x = usable[feature_cols].astype(float).fillna(0.0).to_numpy()
    y_true = usable[label_column].astype(str).to_numpy()

    metrics: dict[str, dict[str, float]] = {}
    pred_frames: list[pd.DataFrame] = []
    reports: dict[str, str] = {}

    for name, model in models.items():
        y_pred = model.predict(x)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        reports[name] = classification_report(y_true, y_pred, zero_division=0)
        pred_df = usable[
            [
                c
                for c in ["relative_path", "file_name", "subject_id", label_column]
                if c in usable.columns
            ]
        ].copy()
        pred_df["model"] = name
        pred_df["predicted"] = y_pred
        pred_df["correct"] = pred_df["predicted"].astype(str) == pred_df[label_column].astype(str)
        pred_frames.append(pred_df)

    out_dir = paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "domain_transfer_predictions.csv"
    report_path = out_dir / "domain_transfer_smartphone.md"

    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(predictions_path, index=False)
    else:
        pd.DataFrame().to_csv(predictions_path, index=False)

    _write_report(
        report_path,
        metrics=metrics,
        cwpv_val_metrics=cwpv_val_metrics,
        n_phone=len(usable),
        label_column=label_column,
        n_features=len(feature_cols),
        reports=reports,
        warnings=warnings,
    )

    return DomainTransferResult(
        metrics=metrics,
        report_path=report_path,
        predictions_path=predictions_path,
        warnings=warnings,
    )


def _write_report(
    path: Path,
    *,
    metrics: dict[str, dict[str, float]],
    cwpv_val_metrics: dict[str, dict[str, float]],
    n_phone: int,
    label_column: str,
    n_features: int,
    reports: dict[str, str],
    warnings: list[str],
) -> None:
    lines = [
        "# Smartphone Domain-Transfer Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Setup:** Models trained on CWPV `train` only. Phone clips are evaluation-only. "
        "CWPV `test` is held out.",
        "",
        f"- Phone videos scored: {n_phone}",
        f"- Feature dimensions: {n_features}",
        f"- Label column: `{label_column}`",
        "",
        "## Metrics comparison",
        "",
        "| Model | CWPV val accuracy | CWPV val macro-F1 | Phone accuracy | Phone macro-F1 | Delta accuracy | Delta macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in metrics.items():
        src = cwpv_val_metrics.get(name) or {}
        src_acc = float(src.get("accuracy", float("nan")))
        src_f1 = float(src.get("macro_f1", float("nan")))
        d_acc = m["accuracy"] - src_acc if np.isfinite(src_acc) else float("nan")
        d_f1 = m["macro_f1"] - src_f1 if np.isfinite(src_f1) else float("nan")
        lines.append(
            f"| {name} | {_fmt(src_acc)} | {_fmt(src_f1)} | "
            f"{m['accuracy']:.4f} | {m['macro_f1']:.4f} | {_fmt(d_acc)} | {_fmt(d_f1)} |"
        )

    if metrics:
        best = max(metrics, key=lambda k: metrics[k].get("macro_f1", 0.0))
        lines.extend(
            [
                "",
                "## Classification report (phone, best by macro-F1)",
                "",
                f"Best model: `{best}`",
                "",
                "```",
                reports[best].rstrip(),
                "```",
            ]
        )

    if warnings:
        lines.extend(["", "## Notes", ""])
        for w in warnings:
            lines.append(f"- {w}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"
