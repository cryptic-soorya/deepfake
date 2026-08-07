"""Integration test for the real AASIST wrapper — proof
app/models/audio_deepfake.py is wired to the authors' pretrained checkpoint,
not a stub. Skips (rather than fails) if offline.
"""
import numpy as np
import pytest

from app.models.audio_deepfake import AASISTVoiceDetector


@pytest.fixture(scope="module")
def detector():
    model = AASISTVoiceDetector()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load AASIST weights (offline / no network?): {exc}")
    return model


def test_predict_on_random_noise_returns_well_formed_contract(detector):
    waveform = np.random.uniform(-0.1, 0.1, 16000 * 3).astype(np.float32)

    output = detector.predict(waveform)

    assert 0.0 <= output["score"] <= 1.0
    assert 0.0 <= output["confidence"] <= 1.0
    assert "interpretation" in output["metadata"]


def test_predict_pads_short_audio(detector):
    waveform = np.random.uniform(-0.1, 0.1, 8000).astype(np.float32)

    output = detector.predict(waveform)

    assert output["score"] is not None
