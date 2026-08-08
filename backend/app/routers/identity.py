import asyncio

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.db import get_db
from app.db_models import Identity
from app.models.face_recognition import ArcFaceEmbedder
from app.models.liveness import MiniFASNetLiveness

router = APIRouter()

MATCH_THRESHOLD = 0.55
LIVENESS_THRESHOLD = 0.5

# Process-wide singletons, loaded at most once in a worker thread so a slow
# first-time load (ONNX/torch init, possible checkpoint download) never blocks
# the event loop -- mirrors app/routers/stream.py's `_ensure_models_loaded`.
_embedder = ArcFaceEmbedder()
_liveness = MiniFASNetLiveness()
_embedder_loaded = False
_liveness_loaded = False
_load_lock = asyncio.Lock()


async def _get_embedder() -> ArcFaceEmbedder:
    global _embedder_loaded
    if not _embedder_loaded:
        async with _load_lock:
            if not _embedder_loaded:
                await asyncio.to_thread(_embedder.load)
                _embedder_loaded = True
    return _embedder


async def _get_liveness() -> MiniFASNetLiveness:
    global _liveness_loaded
    if not _liveness_loaded:
        async with _load_lock:
            if not _liveness_loaded:
                await asyncio.to_thread(_liveness.load)
                _liveness_loaded = True
    return _liveness


@router.post("/enroll")
async def enroll_identity(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Enroll a face embedding for a user (ArcFace)."""
    image_bytes = await file.read()

    try:
        embedder = await _get_embedder()
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="face recognition model not yet integrated")

    result = await asyncio.to_thread(embedder.predict, image_bytes)
    faces = result["raw"]["faces"]
    if not faces:
        raise HTTPException(status_code=422, detail="no face detected in enrollment image")

    best_face = max(faces, key=lambda f: f["det_score"])

    existing = await db.execute(select(Identity).where(Identity.user_id == user_id))
    identity = existing.scalar_one_or_none()
    if identity is None:
        identity = Identity(user_id=user_id, embedding=best_face["embedding"], det_score=best_face["det_score"])
        db.add(identity)
    else:
        identity.embedding = best_face["embedding"]
        identity.det_score = best_face["det_score"]

    await write_audit_log(
        db,
        user_id=user_id,
        action="identity.enroll",
        resource_id=user_id,
        metadata={"det_score": best_face["det_score"], "num_faces_detected": len(faces)},
    )
    await db.commit()

    return {"user_id": user_id, "enrolled": True, "det_score": best_face["det_score"]}


@router.post("/verify")
async def verify_identity(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Verify a live face against an enrolled embedding + liveness check."""
    image_bytes = await file.read()

    result = await db.execute(select(Identity).where(Identity.user_id == user_id))
    identity = result.scalar_one_or_none()
    if identity is None:
        raise HTTPException(status_code=404, detail="no enrolled identity for this user_id")

    try:
        embedder = await _get_embedder()
        liveness_model = await _get_liveness()
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="face recognition/liveness model not yet integrated")

    liveness_result = await asyncio.to_thread(liveness_model.predict, image_bytes)
    is_live = liveness_result["metadata"].get("is_live", False)

    embed_result = await asyncio.to_thread(embedder.predict, image_bytes)
    probe_faces = embed_result["raw"]["faces"]

    match_score = 0.0
    if probe_faces:
        import numpy as np

        probe_face = max(probe_faces, key=lambda f: f["det_score"])
        similarity = float(np.dot(probe_face["embedding"], identity.embedding))
        match_score = (similarity + 1.0) / 2.0

    is_match = match_score >= MATCH_THRESHOLD
    verified = bool(is_match and is_live)

    await write_audit_log(
        db,
        user_id=user_id,
        action="identity.verify",
        resource_id=user_id,
        metadata={
            "match_score": match_score,
            "is_match": is_match,
            "is_live": is_live,
            "liveness_score": liveness_result["score"],
            "verified": verified,
        },
    )
    await db.commit()

    return {
        "user_id": user_id,
        "verified": verified,
        "match_score": match_score,
        "is_match": is_match,
        "liveness": {
            "is_live": is_live,
            "score": liveness_result["score"],
        },
    }
