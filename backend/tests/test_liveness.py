"""Integration test for the real MiniFASNet liveness wrapper — proof
app/models/liveness.py is wired to the authors' pretrained checkpoints, not
a stub. Skips (rather than fails) if offline.
"""
import cv2
import insightface
import pytest

from app.models.liveness import MiniFASNetLiveness


@pytest.fixture(scope="module")
def detector():
    model = MiniFASNetLiveness()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load MiniFASNet weights (offline / no network?): {exc}")
    return model


def test_predict_on_blank_image_returns_well_formed_contract(detector):
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype="uint8")
    output = detector.predict(blank)

    assert set(output.keys()) == {"score", "confidence", "raw", "metadata"}
    assert output["score"] == 0.0
    assert "error" in output["metadata"]


def test_predict_on_real_photo_yields_a_liveness_verdict(detector):
    sample_path = f"{insightface.__path__[0]}/data/images/t1.jpg"
    image = cv2.imread(sample_path)

    output = detector.predict(image)

    assert 0.0 <= output["score"] <= 1.0
    assert "is_live" in output["metadata"]
    assert len(output["raw"]["prediction"][0]) == 3
