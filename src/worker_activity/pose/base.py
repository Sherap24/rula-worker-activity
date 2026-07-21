"""Pose estimation data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoseFrame:
    frame_index: int
    timestamp_seconds: float | None
    landmarks: dict[str, dict[str, float]]
    detection_confidence: float | None = None


@dataclass
class PoseSequence:
    video_relative_path: str
    fps: float | None
    frames: list[PoseFrame] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for frame in self.frames:
            for name, coords in frame.landmarks.items():
                records.append(
                    {
                        "video_relative_path": self.video_relative_path,
                        "frame_index": frame.frame_index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "landmark": name,
                        "x": coords.get("x"),
                        "y": coords.get("y"),
                        "z": coords.get("z"),
                        "visibility": coords.get("visibility"),
                        "detection_confidence": frame.detection_confidence,
                    }
                )
        return records
