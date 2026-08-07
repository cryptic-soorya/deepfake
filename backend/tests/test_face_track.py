"""Unit tests for face tracking/cropping logic.

Uses a fake detector (not real SCRFD weights, which need a network fetch)
so this exercises track_faces' own logic -- holding the last bbox across
detection dropouts, cropping/resizing, the empty-video case -- independent
of face_detector.py's model loading. test_face_detector.py already covers
SCRFD itself.
"""
import cv2
import numpy as np
import pytest

from app.pipeline.face_track import track_faces


class _FakeDetector:
    """Detects a face on every Nth call, based on `should_detect`."""

    def __init__(self, should_detect):
        self._should_detect = iter(should_detect)
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        detect = next(self._should_detect, False)
        if detect:
            return {"raw": {"faces": [{"bbox": [10.0, 10.0, 40.0, 40.0], "det_score": 0.99}]}}
        return {"raw": {"faces": []}}


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (64, 64))
    for i in range(25):  # 1s @ 25fps native -> matches target_fps 1:1
        frame = np.full((64, 64, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_face_detected_every_frame(synthetic_video):
    detector = _FakeDetector(should_detect=[True] * 25)
    result = track_faces(str(synthetic_video), target_fps=25, crop_size=32, detector=detector)

    assert result["num_frames"] == 25
    assert result["num_frames_with_face"] == 25
    assert result["held_frames"] == 0
    assert result["frames"].shape == (25, 32, 32, 3)


def test_holds_last_bbox_across_dropout(synthetic_video):
    # face seen on frame 0, lost for frames 1-2, seen again from frame 3 on
    should_detect = [True] + [False, False] + [True] * 22
    detector = _FakeDetector(should_detect=should_detect)
    result = track_faces(str(synthetic_video), target_fps=25, crop_size=32, detector=detector)

    assert result["num_frames"] == 25  # continuous -- no gaps in output
    assert result["num_frames_with_face"] == 23
    assert result["held_frames"] == 2


def test_no_face_ever_detected_returns_empty(synthetic_video):
    detector = _FakeDetector(should_detect=[False] * 25)
    result = track_faces(str(synthetic_video), target_fps=25, crop_size=32, detector=detector)

    assert result["num_frames"] == 0
    assert result["num_frames_with_face"] == 0
    assert result["frames"].shape == (0, 32, 32, 3)


def test_rejects_unsupported_input_type():
    with pytest.raises(TypeError):
        track_faces(12345, detector=_FakeDetector(should_detect=[]))
