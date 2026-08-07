"""Integration test for app/models/gradcam.py -- proof Grad-CAM actually
runs a forward+backward pass through the real EfficientNet-B4/SBI checkpoint
and produces a heatmap, not a stub. Skips (rather than fails) if offline,
same convention as test_frame_classifier.py.
"""
import base64

import cv2
import insightface
import numpy as np
import pytest

from app.models.gradcam import GradCAMExplainer


@pytest.fixture(scope="module")
def explainer():
    model = GradCAMExplainer()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load SBI/EfficientNet-B4 weights (offline / no network?): {exc}")
    return model


def test_predict_on_real_photo_returns_heatmap(explainer):
    sample_path = f"{insightface.__path__[0]}/data/images/t1.jpg"
    image = cv2.imread(sample_path)

    output = explainer.predict(image)

    assert 0.0 <= output["score"] <= 1.0
    assert 0.0 <= output["confidence"] <= 1.0
    heatmap_b64 = output["metadata"]["heatmap_png_base64"]
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(heatmap_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (380, 380)


def test_predict_on_blank_image_reports_no_face(explainer):
    blank = np.zeros((480, 640, 3), dtype="uint8")

    output = explainer.predict(blank)

    assert output["score"] is None
    assert "error" in output["metadata"]
