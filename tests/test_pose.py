"""Tests for pose data structures and pipeline selection."""

import pandas as pd

from worker_activity.pose.base import PoseFrame, PoseSequence
from worker_activity.pose.pipeline import _select_cwpv_videos


def test_pose_sequence_to_records():
    seq = PoseSequence(
        video_relative_path="cwpv/extracted/0111Camera_1.avi",
        fps=25.0,
        frames=[
            PoseFrame(
                frame_index=0,
                timestamp_seconds=0.0,
                landmarks={"NOSE": {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}},
            )
        ],
    )
    records = seq.to_records()
    assert len(records) == 1
    assert records[0]["landmark"] == "NOSE"
    assert records[0]["x"] == 0.5


def test_select_cwpv_videos_filters():
    inventory = pd.DataFrame(
        [
            {
                "source": "cwpv",
                "file_name": "0931Camera_3.avi",
                "relative_path": "cwpv/extracted/0931Camera_3.avi",
            },
            {
                "source": "cwpv",
                "file_name": "0941Camera_1.avi",
                "relative_path": "cwpv/extracted/0941Camera_1.avi",
            },
            {
                "source": "cml",
                "file_name": "000001.json",
                "relative_path": "cml/extracted/x.json",
            },
        ]
    )
    selected = _select_cwpv_videos(
        inventory,
        subject_id="09",
        motion_id="3",
        camera_id=None,
        max_videos=5,
    )
    assert len(selected) == 1
    assert selected.iloc[0]["file_name"] == "0931Camera_3.avi"
