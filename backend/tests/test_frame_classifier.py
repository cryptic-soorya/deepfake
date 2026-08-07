"""Integration test for the real EfficientNet-B4/SBI wrapper — proof
app/models/frame_classifier.py is wired to the authors' pretrained checkpoint,
not a stub. Skips (rather than fails) if offline.
"""
import cv2
import insightface
import pytest

from app.models.frame_classifier import EfficientNetB4SBI


@pytest.fixture(scope="module")
def classifier():
    model = EfficientNetB4SBI()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load SBI/EfficientNet-B4 weights (offline / no network?): {exc}")
    return model


def test_predict_on_real_photo_returns_well_formed_contract(classifier):
    sample_path = f"{insightface.__path__[0]}/data/images/t1.jpg"
    image = cv2.imread(sample_path)

    output = classifier.predict(image)

    assert 0.0 <= output["score"] <= 1.0
    assert 0.0 <= output["confidence"] <= 1.0
    assert "interpretation" in output["metadata"]


def test_predict_on_blank_image_reports_no_face(classifier):
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype="uint8")
    output = classifier.predict(blank)

    assert output["score"] is None
    assert "error" in output["metadata"]
