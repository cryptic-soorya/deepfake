"""Unit tests for the interim fixed-weighted-average fusion head.

Pure logic, no model weights/network involved -- unlike test_frame_classifier.py
et al., this never needs to skip for being offline.
"""
import pytest

from app.fusion.scoring import fuse


def test_both_modalities_agree_fake():
    out = fuse(
        {
            "frame_classifier": {"score": 0.9, "confidence": 0.9, "raw": {}, "metadata": {}},
            # audio_deepfake score is P(real); 0.1 -> P(fake) 0.9
            "audio_deepfake": {"score": 0.1, "confidence": 0.85, "raw": {}, "metadata": {}},
        }
    )
    assert out["verdict"] == "fake"
    assert out["score"] > 0.7


def test_both_modalities_agree_authentic():
    out = fuse(
        {
            "frame_classifier": {"score": 0.05, "confidence": 0.9, "raw": {}, "metadata": {}},
            "audio_deepfake": {"score": 0.95, "confidence": 0.9, "raw": {}, "metadata": {}},
        }
    )
    assert out["verdict"] == "authentic"
    assert out["score"] < 0.3


def test_image_only_uses_single_modality_score_directly():
    out = fuse({"frame_classifier": {"score": 0.8, "confidence": 0.8, "raw": {}, "metadata": {}}})
    assert out["score"] == 0.8
    assert out["verdict"] == "fake"


def test_modality_with_no_score_is_excluded_not_zeroed():
    out = fuse(
        {
            "frame_classifier": {"score": None, "confidence": None, "raw": None, "metadata": {"error": "no face detected"}},
            "audio_deepfake": {"score": 0.2, "confidence": 0.7, "raw": {}, "metadata": {}},
        }
    )
    assert "frame_classifier" in out["metadata"]["skipped"]
    # fused score should equal the surviving modality's own P(fake), not be
    # dragged toward zero by treating the missing one as "authentic".
    assert out["score"] == pytest.approx(0.8)


def test_all_modalities_missing_score_is_inconclusive_not_blocked():
    out = fuse(
        {
            "frame_classifier": {"score": None, "confidence": None, "raw": None, "metadata": {}},
            "audio_deepfake": {"score": None, "confidence": None, "raw": None, "metadata": {}},
        }
    )
    assert out["score"] is None
    assert out["verdict"] == "inconclusive"


def test_suspicious_band_between_thresholds():
    out = fuse({"frame_classifier": {"score": 0.5, "confidence": 0.5, "raw": {}, "metadata": {}}})
    assert out["verdict"] == "suspicious"


def test_all_three_modalities_agree_fake_including_lipsync():
    out = fuse(
        {
            "frame_classifier": {"score": 0.85, "confidence": 0.9, "raw": {}, "metadata": {}},
            "audio_deepfake": {"score": 0.15, "confidence": 0.85, "raw": {}, "metadata": {}},
            # lipsync score is A/V sync confidence; 0.1 -> desynced -> P(fake) 0.9
            "lipsync": {"score": 0.1, "confidence": 0.1, "raw": {}, "metadata": {}},
        }
    )
    assert out["verdict"] == "fake"
    assert out["metadata"]["per_model_fake_prob"]["lipsync"] == pytest.approx(0.9)


def test_lipsync_alone_desynced_flags_fake():
    out = fuse({"lipsync": {"score": 0.05, "confidence": 0.05, "raw": {}, "metadata": {}}})
    assert out["score"] == pytest.approx(0.95)
    assert out["verdict"] == "fake"


def test_lipsync_no_face_tracked_is_excluded_like_any_missing_score():
    out = fuse(
        {
            "frame_classifier": {"score": 0.6, "confidence": 0.6, "raw": {}, "metadata": {}},
            "audio_deepfake": {"score": 0.5, "confidence": 0.5, "raw": {}, "metadata": {}},
            "lipsync": {"score": None, "confidence": None, "raw": None, "metadata": {"status": "no_face_tracked_in_video"}},
        }
    )
    assert "lipsync" in out["metadata"]["skipped"]
    assert "lipsync" not in out["metadata"]["per_model_fake_prob"]
