"""Unit tests for fixed-rate video frame sampling.

Builds a small synthetic .mp4 on the fly with cv2.VideoWriter rather than
shipping a binary fixture -- no network, no skip-if-offline needed.
"""
import cv2
import numpy as np
import pytest

from app.pipeline.frame_sampler import sample_frames


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    for i in range(50):  # 5s @ 10fps native
        frame = np.full((64, 64, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_sample_frames_from_path_at_target_rate(synthetic_video):
    frames = sample_frames(str(synthetic_video), fps=2.0)
    # 10fps native, 2fps target -> every 5th frame, 50 frames -> 10 samples
    assert len(frames) == 10
    assert frames[0]["frame"].shape == (64, 64, 3)
    assert [f["index"] for f in frames] == list(range(10))


def test_sample_frames_from_bytes(synthetic_video):
    data = synthetic_video.read_bytes()
    frames = sample_frames(data, fps=2.0)
    assert len(frames) == 10


def test_sample_frames_respects_max_frames_cap(synthetic_video):
    frames = sample_frames(str(synthetic_video), fps=10.0, max_frames=3)
    assert len(frames) == 3


def test_sample_frames_rejects_invalid_fps(synthetic_video):
    with pytest.raises(ValueError):
        sample_frames(str(synthetic_video), fps=0)


def test_sample_frames_rejects_unsupported_input_type():
    with pytest.raises(TypeError):
        sample_frames(12345)


def test_sample_frames_raises_on_unreadable_video(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not video data")
    with pytest.raises(ValueError):
        sample_frames(str(bogus))
