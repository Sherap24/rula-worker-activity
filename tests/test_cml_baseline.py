"""Tests for CML baseline mapping, subjects, representations, and splits."""

from __future__ import annotations

import pandas as pd
import pytest

from worker_activity.data.cml_baseline import (
    LeakageAuditError,
    audit_leakage,
    build_cml_splits,
    enrich_cml_inventory_rows,
)
from worker_activity.data.cml_label_map import apply_label_mapping, load_cml_label_map
from worker_activity.data.cml_representation import (
    assign_representation_fields,
    audit_representations,
    build_logical_sample_id,
)
from worker_activity.data.cml_subject import parse_subject_id


@pytest.fixture
def label_map(repo_root):
    return load_cml_label_map(repo_root / "configs" / "label_map_cml.yaml")


def test_ambiguous_stand_up_maps_to_other_excluded(label_map):
    result = apply_label_mapping("stand up", label_map)
    assert result["canonical_activity"] == "other"
    assert result["include_in_baseline"] is False


def test_ambiguous_sitting_maps_to_other_excluded(label_map):
    result = apply_label_mapping("sitting", label_map)
    assert result["canonical_activity"] == "other"
    assert result["include_in_baseline"] is False


def test_picking_up_provisional_baseline(label_map):
    result = apply_label_mapping("picking up", label_map)
    assert result["canonical_activity"] == "lifting_lowering"
    assert result["include_in_baseline"] is True
    assert result["mapping_status"] == "provisional"


def test_crawling_not_kneeling(label_map):
    result = apply_label_mapping("crawling", label_map)
    assert result["canonical_activity"] == "other"
    assert result["include_in_baseline"] is False


def test_running_excluded_from_baseline(label_map):
    result = apply_label_mapping("running", label_map)
    assert result["canonical_activity"] == "other"
    assert result["include_in_baseline"] is False


def test_calibration_excluded(label_map):
    result = apply_label_mapping("dynamic calibration", label_map)
    assert result["mapping_status"] == "calibration_only"
    assert result["include_in_baseline"] is False


def test_cmu_subject_parsing():
    result = parse_subject_id("CMU", "bend/115_01_01.txt")
    assert result.subject_id == "CMU:115"
    assert result.subject_parse_status == "parsed_verified_pattern"


def test_cmu_subject_namespaced():
    result = parse_subject_id("CMU", "bend/115_01_01.txt")
    assert result.subject_id.startswith("CMU:")


def test_unknown_source_unresolved():
    result = parse_subject_id("UnknownDB", "foo/bar.txt")
    assert result.subject_id is None
    assert result.subject_parse_status == "unresolved_pattern"


def test_berkeley_subject_parsing():
    result = parse_subject_id("Berkeley", "bending/skl_s03_a03_r01.txt")
    assert result.subject_id == "Berkeley:s03"


def test_logical_sample_id_stable():
    a = build_logical_sample_id("CMU", "bend/115_01_01.txt", "bending")
    b = build_logical_sample_id("CMU", "bend/115_01_01.txt", "bending")
    assert a == b
    assert a is not None
    assert "15" not in a and "20" not in a


def test_representation_pairing():
    rows = []
    base = {
        "source": "cml",
        "source_dataset": "CMU",
        "source_file": "bend/115_01_01.txt",
        "raw_activity_label": "bending",
        "construction_subset": True,
    }
    for layout, view in [(15, "15_nodes"), (20, "20_nodes")]:
        row = base.copy()
        row["view_id"] = view
        row["skeleton_layout"] = layout
        rows.append(row)
    df = assign_representation_fields(pd.DataFrame(rows))
    assert df["logical_sample_id"].nunique() == 1
    assert set(df["skeleton_layout"]) == {15, 20}


def test_representation_audit_ok():
    rows = []
    base = {
        "source_dataset": "CMU",
        "source_file": "bend/115_01_01.txt",
        "raw_activity_label": "bending",
    }
    for view in ["15_nodes", "20_nodes"]:
        row = base.copy()
        row["view_id"] = view
        rows.append(row)
    df = assign_representation_fields(pd.DataFrame(rows))
    audit_df, summary = audit_representations(df)
    assert summary["ok_pairs"] == 1


def _synthetic_cml_inventory() -> pd.DataFrame:
    """Minimal construction subset with two subjects and baseline-eligible labels."""
    rows = []
    subjects = [
        ("CMU", "bend/101_01_01.txt", "bending"),
        ("CMU", "bend/102_01_01.txt", "bending"),
        ("CMU", "walk/103_01_01.txt", "walking"),
        ("CMU", "walk/104_01_01.txt", "walking"),
        ("CMU", "walk/105_01_01.txt", "walking"),
    ]
    for ds, sf, label in subjects:
        for view in ["15_nodes", "20_nodes"]:
            rows.append(
                {
                    "source": "cml",
                    "source_type": "skeleton",
                    "relative_path": f"cml/extracted/Construction_Related_Data/x/{view}/{label}/000001.json",
                    "file_name": "000001.json",
                    "extension": ".json",
                    "video_id": sf,
                    "clip_id": "000001",
                    "view_id": view,
                    "raw_activity_label": label,
                    "notes": f"subset=construction_related; data_source={ds}",
                    "integrity_status": "ok",
                    "metadata_status": "extracted",
                }
            )
    return pd.DataFrame(rows)


def test_enrich_preserves_raw_label(label_map):
    df = _synthetic_cml_inventory()
    raw = df["raw_activity_label"].copy()
    enriched = enrich_cml_inventory_rows(df, label_map)
    pd.testing.assert_series_equal(enriched["raw_activity_label"], raw, check_names=False)


def test_baseline_excludes_other_labels(label_map):
    df = _synthetic_cml_inventory()
    extra = {
        "source": "cml",
        "source_type": "skeleton",
        "relative_path": "cml/extracted/Construction_Related_Data/x/15_nodes/running/000001.json",
        "file_name": "000001.json",
        "extension": ".json",
        "video_id": "run/106_01_01.txt",
        "clip_id": "000001",
        "view_id": "15_nodes",
        "raw_activity_label": "running",
        "notes": "subset=construction_related; data_source=CMU",
        "integrity_status": "ok",
        "metadata_status": "extracted",
    }
    df = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)
    enriched = enrich_cml_inventory_rows(df, label_map)
    manifests, _ = build_cml_splits(enriched)
    assert "running" not in manifests["train"]["raw_activity_label"].values
    assert manifests["excluded"]["raw_activity_label"].eq("running").any()


def test_paired_representations_same_split(label_map):
    df = _synthetic_cml_inventory()
    enriched = enrich_cml_inventory_rows(df, label_map)
    manifests, _ = build_cml_splits(enriched)
    train = manifests["train"]
    for logical_id in train["logical_sample_id"].unique():
        all_reps = enriched[enriched["logical_sample_id"] == logical_id]
        train_reps = train[train["logical_sample_id"] == logical_id]
        assert len(train_reps) == len(all_reps[all_reps["include_in_baseline"] == True])


def test_subject_disjoint_splits(label_map):
    df = _synthetic_cml_inventory()
    enriched = enrich_cml_inventory_rows(df, label_map)
    manifests, _ = build_cml_splits(enriched)
    train_subjects = set(manifests["train"]["subject_id"])
    val_subjects = set(manifests["validation"]["subject_id"])
    test_subjects = set(manifests["test"]["subject_id"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)


def test_unresolved_subjects_excluded_from_val_test(label_map):
    df = _synthetic_cml_inventory()
    bad = {
        "source": "cml",
        "source_type": "skeleton",
        "relative_path": "cml/extracted/Construction_Related_Data/x/15_nodes/walking/x.json",
        "file_name": "x.json",
        "extension": ".json",
        "video_id": None,
        "clip_id": "x",
        "view_id": "15_nodes",
        "raw_activity_label": "walking",
        "notes": "subset=construction_related; data_source=CMU",
        "integrity_status": "ok",
        "metadata_status": "extracted",
    }
    df = pd.concat([df, pd.DataFrame([bad, bad])], ignore_index=True)
    enriched = enrich_cml_inventory_rows(df, label_map)
    manifests, _ = build_cml_splits(enriched)
    assert len(manifests["unresolved_subject"]) >= 2
    assert len(manifests["validation"]) == 0 or manifests["validation"]["subject_id"].notna().all()


def test_leakage_audit_passes(label_map):
    df = _synthetic_cml_inventory()
    enriched = enrich_cml_inventory_rows(df, label_map)
    manifests, _ = build_cml_splits(enriched)
    _, violations = audit_leakage(manifests)
    assert violations == []


def test_leakage_audit_fails_on_overlap():
    row = {
        "subject_id": "CMU:1",
        "logical_sample_id": "cml_abc",
        "representation_group_id": "cml_abc",
        "include_in_baseline": True,
        "raw_activity_label": "walking",
        "canonical_activity": "walking",
    }
    train = pd.DataFrame([row, row])
    val = pd.DataFrame([row])
    test = pd.DataFrame([{**row, "logical_sample_id": "cml_def", "representation_group_id": "cml_def"}])
    manifests = {
        "train": train,
        "validation": val,
        "test": test,
        "unresolved_subject": pd.DataFrame(),
        "excluded": pd.DataFrame(),
    }
    _, violations = audit_leakage(manifests)
    assert any("overlap" in v for v in violations)


def test_deterministic_split_reproduction(label_map):
    df = _synthetic_cml_inventory()
    enriched = enrich_cml_inventory_rows(df, label_map)
    m1, _ = build_cml_splits(enriched)
    m2, _ = build_cml_splits(enriched)
    assert set(m1["train"]["subject_id"]) == set(m2["train"]["subject_id"])
