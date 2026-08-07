from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/")
async def create_scan(file: UploadFile = File(...)):
    """Accept a video/image upload and enqueue it for the inference pipeline."""
    raise NotImplementedError


@router.get("/{scan_id}")
async def get_scan(scan_id: str):
    """Return the fused verdict + explanation for a completed scan."""
    raise NotImplementedError
