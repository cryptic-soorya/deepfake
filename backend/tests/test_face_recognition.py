"""Integration test for the real ArcFace wrapper — proof app/models/face_recognition.py
is wired to real weights, not a stub. Skips (rather than fails) if offline.
"""
import cv2
import insightface
import pytest

from app.models.face_recognition import ArcFaceEmbedder


@pytest.fixture(scope="module")
def embedder():
    model = ArcFaceEmbedder(model_pack="buffalo_sc")
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load ArcFace weights (offline / no network?): {exc}")
    return model


@pytest.fixture(scope="module")
def sample_image():
    sample_path = f"{insightface.__path__[0]}/data/images/t1.jpg"
    return cv2.imread(sample_path)


def test_enroll_returns_well_formed_contract(embedder, sample_image):
    output = embedder.predict(sample_image)

    assert set(output.keys()) == {"score", "confidence", "raw", "metadata"}
    assert output["metadata"]["mode"] == "enroll"
    assert output["metadata"]["num_faces"] > 0
    for face in output["raw"]["faces"]:
        assert len(face["embedding"]) == 512


def test_verify_same_image_is_a_near_perfect_match(embedder, sample_image):
    output = embedder.predict((sample_image, sample_image))

    assert output["metadata"]["mode"] == "verify"
    assert output["score"] > 0.99


def test_verify_returns_error_metadata_when_no_face_present(embedder):
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype="uint8")
    output = embedder.predict((blank, blank))

    assert output["score"] == 0.0
    assert "error" in output["metadata"]
