"""Integration test for the real SCRFD wrapper.

This downloads pretrained InsightFace weights on first run (cached under
~/.insightface/models afterwards) and runs genuine ONNX inference — it is
the proof that app/models/face_detector.py is wired to real weights, not a
stub. Skips (rather than fails) if no network is available, since CI/offline
dev boxes shouldn't be blocked on a model download.
"""
import numpy as np
import pytest

from app.models.face_detector import SCRFDFaceDetector


@pytest.fixture(scope="module")
def detector():
    # buffalo_sc is InsightFace's lightweight pack (SCRFD-500M + MobileFaceNet,
    # ~16MB) — used here only to keep this test's download fast. The wrapper's
    # production default is buffalo_l (see face_detector.py), matching the
    # pack CLAUDE.md specifies for ArcFace so the two share one download.
    model = SCRFDFaceDetector(model_pack="buffalo_sc")
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load SCRFD weights (offline / no network?): {exc}")
    return model


def test_load_produces_a_working_model(detector):
    assert detector._loaded is True


def test_predict_on_blank_image_returns_well_formed_contract(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    output = detector.predict(blank)

    assert set(output.keys()) == {"score", "confidence", "raw", "metadata"}
    assert output["metadata"]["num_faces"] == 0
    assert output["raw"]["faces"] == []
    assert output["score"] == 0.0


def test_predict_accepts_encoded_image_bytes(detector):
    import cv2

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok

    output = detector.predict(encoded.tobytes())

    assert output["metadata"]["image_shape"] == [480, 640, 3]


def test_predict_detects_real_faces_in_sample_photo(detector):
    """Proves this is genuine detection, not just a well-shaped empty result:
    runs against the group photo InsightFace ships as its own test fixture
    (insightface/data/images/t1.jpg) rather than an externally sourced image.
    """
    import cv2
    import insightface

    sample_path = f"{insightface.__path__[0]}/data/images/t1.jpg"
    image = cv2.imread(sample_path)

    output = detector.predict(image)

    assert output["metadata"]["num_faces"] > 0
    assert output["score"] > 0.5
    for face in output["raw"]["faces"]:
        assert len(face["bbox"]) == 4
        assert len(face["landmarks_5pt"]) == 5
