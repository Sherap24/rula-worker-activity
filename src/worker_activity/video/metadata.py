"""Video metadata extraction — OpenCV baseline with optional ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


@dataclass
class VideoMetadata:
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    width: int | None
    height: int | None
    codec: str | None
    status: str
    provider: str
    notes: str | None = None


def extract_video_metadata(
    path: Path,
    *,
    provider: str = "opencv",
    ffprobe_binary: str = "ffprobe",
) -> VideoMetadata:
    if provider == "ffprobe" and shutil.which(ffprobe_binary):
        try:
            return _extract_ffprobe(path, ffprobe_binary)
        except OSError as exc:
            return VideoMetadata(
                fps=None,
                frame_count=None,
                duration_seconds=None,
                width=None,
                height=None,
                codec=None,
                status="failed",
                provider="ffprobe",
                notes=f"ffprobe failed, falling back unavailable: {exc}",
            )
    return _extract_opencv(path)


def _extract_opencv(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return VideoMetadata(
            fps=None,
            frame_count=None,
            duration_seconds=None,
            width=None,
            height=None,
            codec=None,
            status="failed",
            provider="opencv",
            notes="OpenCV could not open video",
        )
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip() or None

        fps_val = float(fps) if fps and fps > 0 else None
        duration = None
        if fps_val and frame_count > 0:
            duration = frame_count / fps_val

        return VideoMetadata(
            fps=fps_val,
            frame_count=frame_count if frame_count > 0 else None,
            duration_seconds=duration,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
            codec=codec,
            status="extracted",
            provider="opencv",
        )
    finally:
        capture.release()


def _extract_ffprobe(path: Path, binary: str) -> VideoMetadata:
    cmd = [
        binary,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    video_stream = next(
        (s for s in payload.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    fps = None
    if video_stream.get("avg_frame_rate") and video_stream["avg_frame_rate"] != "0/0":
        num, den = video_stream["avg_frame_rate"].split("/")
        if float(den) != 0:
            fps = float(num) / float(den)
    duration = None
    if video_stream.get("duration"):
        duration = float(video_stream["duration"])
    elif payload.get("format", {}).get("duration"):
        duration = float(payload["format"]["duration"])
    frame_count = None
    if video_stream.get("nb_frames"):
        frame_count = int(video_stream["nb_frames"])
    elif fps and duration:
        frame_count = int(round(fps * duration))

    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        width=int(video_stream["width"]) if video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream.get("height") else None,
        codec=video_stream.get("codec_name"),
        status="extracted",
        provider="ffprobe",
    )
