"""Tests for video metadata extraction."""

from pathlib import Path

from worker_activity.video.metadata import extract_video_metadata


def test_opencv_metadata_on_synthetic_video(synthetic_video: Path):
    meta = extract_video_metadata(synthetic_video, provider="opencv")
    assert meta.status == "extracted"
    assert meta.provider == "opencv"
    assert meta.width == 64
    assert meta.height == 48
    assert meta.frame_count is not None
    assert meta.frame_count >= 1


def test_missing_video_returns_failed(tmp_path: Path):
    meta = extract_video_metadata(tmp_path / "missing.mp4", provider="opencv")
    assert meta.status == "failed"
