"""Tests for subject-disjoint splitting."""

import pandas as pd

from worker_activity.data.splits import subject_disjoint_split


def _make_inventory(subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": [f"clip_{i}" for i in range(len(subjects))],
            "subject_id": subjects,
        }
    )


def test_subject_disjoint_no_overlap():
    df = _make_inventory(["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"])
    result = subject_disjoint_split(df, seed=42)
    train_subjects = set(result.train["subject_id"])
    val_subjects = set(result.val["subject_id"])
    test_subjects = set(result.test["subject_id"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_missing_subject_fallback_warning():
    df = pd.DataFrame({"clip_id": ["a", "b"], "subject_id": [None, None]})
    result = subject_disjoint_split(df, seed=1)
    assert result.method == "clip_disjoint"
    assert any("clip-disjoint" in w.lower() for w in result.warnings)


def test_empty_inventory_split():
    df = pd.DataFrame()
    result = subject_disjoint_split(df)
    assert result.method == "empty"
