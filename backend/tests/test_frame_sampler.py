"""Unit test for the video frame sampler used by frame_classifier_worker."""
import cv2
import numpy as np
import pytest

from app.pipeline.frame_sampler import sample_frames


@pytest.fixture
def synthetic_video(tmp_path):
    """A 2-second, 10fps synthetic video (20 frames) with a distinct color per frame."""
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(20):
        frame = np.full((32, 32, 3), fill_value=i * 10 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_sample_frames_downsamples_to_requested_rate(synthetic_video):
    # 10fps source, sampled at 2fps -> every 5th frame -> 4 frames from 20.
    frames = list(sample_frames(synthetic_video, fps=2.0))
    assert len(frames) == 4
    assert all(frame.shape == (32, 32, 3) for frame in frames)


def test_sample_frames_at_source_rate_yields_every_frame(synthetic_video):
    frames = list(sample_frames(synthetic_video, fps=10.0))
    assert len(frames) == 20


def test_sample_frames_missing_file_raises():
    with pytest.raises(ValueError):
        list(sample_frames("/no/such/video.mp4"))
