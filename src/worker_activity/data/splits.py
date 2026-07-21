"""Train/validation/test splitting with subject-disjoint policy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    warnings: list[str]
    method: str


def load_splits_config(path: Path | None = None) -> dict[str, Any]:
    root = find_repo_root()
    splits_path = path or root / "configs" / "splits.yaml"
    return load_yaml(splits_path)


def subject_disjoint_split(
    df: pd.DataFrame,
    *,
    subject_column: str = "subject_id",
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    allow_clip_disjoint_fallback: bool = True,
    min_clips_per_subject: int = 1,
) -> SplitResult:
    """Split inventory rows by subject when possible."""
    ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    warnings: list[str] = []

    if df.empty:
        return SplitResult(
            train=df.copy(),
            val=df.copy(),
            test=df.copy(),
            warnings=["Empty inventory — no rows to split."],
            method="empty",
        )

    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    if subject_column not in df.columns:
        warnings.append(f"Column '{subject_column}' missing — using clip-disjoint fallback.")
        return _clip_disjoint_split(df, seed=seed, ratios=ratios, warnings=warnings)

    subjects_series = df[subject_column]
    has_subject = subjects_series.notna() & (subjects_series.astype(str).str.strip() != "")
    if not has_subject.any():
        warnings.append(
            f"No usable '{subject_column}' values — using clip-disjoint fallback."
        )
        if not allow_clip_disjoint_fallback:
            raise ValueError("Subject IDs required but missing.")
        return _clip_disjoint_split(df, seed=seed, ratios=ratios, warnings=warnings)

    if has_subject.sum() < len(df):
        warnings.append(
            f"{len(df) - int(has_subject.sum())} rows lack subject_id; "
            "they will be assigned via clip-disjoint fallback."
        )

    subject_df = df[has_subject].copy()
    unassigned_df = df[~has_subject].copy()

    subject_counts = subject_df.groupby(subject_column).size()
    eligible_subjects = subject_counts[subject_counts >= min_clips_per_subject].index.tolist()
    if not eligible_subjects:
        warnings.append("No subjects meet min_clips_per_subject — clip-disjoint fallback.")
        return _clip_disjoint_split(df, seed=seed, ratios=ratios, warnings=warnings)

    rng = random.Random(seed)
    shuffled_subjects = eligible_subjects.copy()
    rng.shuffle(shuffled_subjects)

    n = len(shuffled_subjects)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train : n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val :])

    train = subject_df[subject_df[subject_column].isin(train_subjects)]
    val = subject_df[subject_df[subject_column].isin(val_subjects)]
    test = subject_df[subject_df[subject_column].isin(test_subjects)]

    if not unassigned_df.empty:
        extra = _clip_disjoint_split(unassigned_df, seed=seed, ratios=ratios, warnings=[])
        train = pd.concat([train, extra.train], ignore_index=True)
        val = pd.concat([val, extra.val], ignore_index=True)
        test = pd.concat([test, extra.test], ignore_index=True)

    return SplitResult(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        warnings=warnings,
        method="subject_disjoint",
    )


def _clip_disjoint_split(
    df: pd.DataFrame,
    *,
    seed: int,
    ratios: dict[str, float],
    warnings: list[str],
) -> SplitResult:
    warnings = list(warnings)
    warnings.append("Using clip-disjoint split (no subject grouping).")
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return SplitResult(
        train=df.loc[train_idx].reset_index(drop=True),
        val=df.loc[val_idx].reset_index(drop=True),
        test=df.loc[test_idx].reset_index(drop=True),
        warnings=warnings,
        method="clip_disjoint",
    )


def write_split_manifests(
    result: SplitResult,
    output_dir: Path,
    filenames: dict[str, str] | None = None,
) -> dict[str, Path]:
    filenames = filenames or {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, frame in [("train", result.train), ("val", result.val), ("test", result.test)]:
        out = output_dir / filenames[key]
        frame.to_csv(out, index=False)
        paths[key] = out
    return paths
