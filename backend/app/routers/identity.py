from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/enroll")
async def enroll_identity(file: UploadFile = File(...)):
    """Enroll a face embedding for a user (ArcFace)."""
    raise NotImplementedError


@router.post("/verify")
async def verify_identity(file: UploadFile = File(...)):
    """Verify a live face against an enrolled embedding + liveness check."""
    raise NotImplementedError
