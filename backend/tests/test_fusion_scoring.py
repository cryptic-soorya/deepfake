"""Unit tests for the interim equal-weighted fusion head."""
import pytest

from app.fusion.scoring import fuse


def _output(score, confidence=0.9):
    return {"score": score, "confidence": confidence, "raw": None, "metadata": {}}


def test_fuse_both_modalities_agree_fake():
    # frame_classifier score is P(fake); audio_deepfake score is P(real/bonafide).
    result = fuse({
        "frame_classifier": _output(0.9),
        "audio_deepfake": _output(0.1),  # low P(real) -> high P(fake) after inversion
    })
    assert result["verdict"] == "fake"
    assert result["score"] == pytest.approx(0.9)
    assert set(result["metadata"]["contributing_models"]) == {"frame_classifier", "audio_deepfake"}


def test_fuse_both_modalities_agree_authentic():
    result = fuse({
        "frame_classifier": _output(0.05),
        "audio_deepfake": _output(0.95),  # high P(real) -> low P(fake)
    })
    assert result["verdict"] == "authentic"
    assert result["score"] == pytest.approx(0.05)


def test_fuse_skips_modalities_with_no_score():
    result = fuse({
        "frame_classifier": _output(0.8),
        "audio_deepfake": {"score": None, "confidence": None, "raw": None, "metadata": {"status": "blocked"}},
    })
    assert result["metadata"]["contributing_models"] == ["frame_classifier"]
    assert result["score"] == pytest.approx(0.8)


def test_fuse_raises_when_no_modality_has_a_score():
    with pytest.raises(NotImplementedError):
        fuse({
            "frame_classifier": {"score": None, "confidence": None, "raw": None, "metadata": {}},
        })
