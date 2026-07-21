"""Tests for CWPV filename parsing and label mapping."""

from pathlib import Path

from worker_activity.data.cwpv_label_map import apply_cwpv_label_mapping, load_cwpv_label_map
from worker_activity.data.cwpv_metadata import enrich_cwpv_row, parse_cwpv_filename


def test_parse_cwpv_filename_example():
    parsed = parse_cwpv_filename("0931Camera_3.avi")
    assert parsed is not None
    assert parsed.participant_id == "09"
    assert parsed.motion_id == "3"
    assert parsed.trial_id == "1"
    assert parsed.camera_id == "3"
    assert parsed.raw_motion_label == "M3_shovel_semi_squat"


def test_parse_cwpv_filename_five_digit_extracted():
    parsed = parse_cwpv_filename("01112Camera_3.avi")
    assert parsed is not None
    assert parsed.participant_id == "01"
    assert parsed.motion_id == "1"
    assert parsed.block_id == "1"
    assert parsed.trial_id == "2"
    assert parsed.camera_id == "3"


def test_parse_cwpv_filename_five_digit_block_two():
    parsed = parse_cwpv_filename("01121Camera_1.avi")
    assert parsed is not None
    assert parsed.participant_id == "01"
    assert parsed.motion_id == "1"
    assert parsed.block_id == "2"
    assert parsed.trial_id == "1"
    assert parsed.camera_id == "1"


def test_parse_cwpv_filename_invalid():
    assert parse_cwpv_filename("random_video.mp4") is None


def test_enrich_cwpv_row(tmp_path: Path):
    video = tmp_path / "0111Camera_1.avi"
    video.write_bytes(b"fake")
    row = enrich_cwpv_row(
        {"notes": None, "metadata_status": "pending"},
        video,
    )
    assert row["subject_id"] == "01"
    assert row["view_id"] == "camera_1"
    assert row["repetition_id"] == "1"
    assert row["raw_activity_label"] == "M1_overhead_hammer_standing"


def test_cwpv_label_map_motion_three(repo_root: Path):
    label_map = load_cwpv_label_map(repo_root / "configs" / "label_map_cwpv.yaml")
    mapped = apply_cwpv_label_mapping("3", label_map)
    assert mapped["canonical_activity"] == "squatting"
    assert mapped["include_in_baseline"] is True
