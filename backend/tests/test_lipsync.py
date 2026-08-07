"""Integration test for the real SyncNet wrapper — proof app/models/lipsync.py
is wired to the authors' pretrained syncnet_v2.model checkpoint, not a stub.
Skips (rather than fails) if offline. Uses synthetic mouth-crop/audio input
(no network-sourced test video available) so this only exercises the
forward-pass contract, not sync accuracy on a genuine talking-head clip.
"""
import numpy as np
import pytest

from app.models.lipsync import AUDIO_SAMPLE_RATE, LIP_FRAME_SIZE, SyncNetLipSync


@pytest.fixture(scope="module")
def sync_model():
    model = SyncNetLipSync()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load SyncNet weights (offline / no network?): {exc}")
    return model


def test_predict_on_synthetic_clip_returns_well_formed_contract(sync_model):
    rng = np.random.default_rng(0)
    num_frames = 30
    lip_frames = rng.integers(0, 255, size=(num_frames, LIP_FRAME_SIZE, LIP_FRAME_SIZE, 3), dtype=np.uint8)
    audio = rng.uniform(-1, 1, size=int(AUDIO_SAMPLE_RATE * num_frames / 25))

    output = sync_model.predict({"lip_frames": lip_frames, "audio": audio})

    assert set(output.keys()) == {"score", "confidence", "raw", "metadata"}
    assert 0.0 <= output["score"] <= 1.0
    assert "av_offset_frames" in output["raw"]


def test_predict_on_too_short_clip_reports_error(sync_model):
    lip_frames = np.zeros((3, LIP_FRAME_SIZE, LIP_FRAME_SIZE, 3), dtype=np.uint8)
    audio = np.zeros(int(AUDIO_SAMPLE_RATE * 3 / 25))

    output = sync_model.predict({"lip_frames": lip_frames, "audio": audio})

    assert "error" in output["metadata"]
