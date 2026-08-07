"""Integration test for the real MediaPipe Face Mesh wrapper — proof
app/models/face_mesh.py is wired to Google's hosted Face Landmarker
checkpoint, not a stub. Skips (rather than fails) if offline.
"""
import cv2
import insightface
import numpy as np
import pytest

from app.models.face_detector import SCRFDFaceDetector
from app.models.face_mesh import MediaPipeFaceMesh


@pytest.fixture(scope="module")
def mesh_model():
    model = MediaPipeFaceMesh()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load Face Landmarker weights (offline / no network?): {exc}")
    return model


@pytest.fixture(scope="module")
def single_face_crop():
    sample_path = f"{insightface.__path__[0]}/data/images/t1.jpg"
    image = cv2.imread(sample_path)

    det = SCRFDFaceDetector(model_pack="buffalo_sc")
    try:
        det.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load SCRFD weights (offline / no network?): {exc}")

    faces = det.predict(image)["raw"]["faces"]
    face = max(faces, key=lambda f: f["det_score"])
    x0, y0, x1, y1 = [int(v) for v in face["bbox"]]
    pad = 20
    h, w = image.shape[:2]
    return image[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)]


def test_predict_on_blank_image_returns_well_formed_contract(mesh_model):
    blank = np.zeros((480, 640, 3), dtype="uint8")

    output = mesh_model.predict(blank)

    assert set(output.keys()) == {"score", "confidence", "raw", "metadata"}
    assert output["metadata"]["num_faces"] == 0


def test_predict_on_cropped_face_returns_478_landmarks(mesh_model, single_face_crop):
    output = mesh_model.predict(single_face_crop)

    assert output["metadata"]["num_faces"] == 1
    assert output["metadata"]["num_landmarks"] == 478
    assert output["raw"]["faces"][0]["blendshapes"]
