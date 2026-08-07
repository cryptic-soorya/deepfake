import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.db import get_db
from app.db_models import Scan, ScanResult, ScanStatus
from app.queue import enqueue
from app.storage import download_media, upload_media

router = APIRouter()


def _media_type(content_type: str | None) -> str:
    if content_type and content_type.startswith("video/"):
        return "video"
    return "image"


@router.post("/")
async def create_scan(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Accept a video/image upload and enqueue it for the inference pipeline."""
    scan_id = str(uuid.uuid4())
    media_type = _media_type(file.content_type)
    media_key = f"scans/{scan_id}/{file.filename}"

    upload_media(media_key, file.file, content_type=file.content_type)

    scan = Scan(id=scan_id, media_key=media_key, media_type=media_type, status=ScanStatus.PENDING.value)
    db.add(scan)
    await write_audit_log(
        db,
        user_id=None,
        action="scan.create",
        resource_id=scan_id,
        metadata={"media_type": media_type, "filename": file.filename},
    )
    await db.commit()

    await enqueue("scan.frames", {"scan_id": scan_id, "media_key": media_key, "media_type": media_type})
    if media_type == "video":
        await enqueue("scan.audio", {"scan_id": scan_id, "media_key": media_key, "media_type": media_type})
        await enqueue("scan.lipsync", {"scan_id": scan_id, "media_key": media_key, "media_type": media_type})

    return {"scan_id": scan_id, "status": ScanStatus.PENDING.value}


@router.get("/{scan_id}")
async def get_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Return the fused verdict + explanation for a completed scan."""
    scan = await db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    result = await db.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))
    results = result.scalars().all()

    return {
        "scan_id": scan.id,
        "status": scan.status,
        "media_type": scan.media_type,
        "fused_score": scan.fused_score,
        "fused_verdict": scan.fused_verdict,
        "explanation": scan.explanation,
        "error": scan.error,
        "model_results": [
            {
                "model_name": r.model_name,
                "score": r.score,
                "confidence": r.confidence,
                "metadata": r.result_metadata,
            }
            for r in results
        ],
    }


@router.get("/{scan_id}/heatmap")
async def get_scan_heatmap(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Return the Grad-CAM heatmap PNG for a scan, proxied from object storage."""
    result = await db.execute(
        select(ScanResult).where(ScanResult.scan_id == scan_id, ScanResult.model_name == "gradcam")
    )
    gradcam_result = result.scalars().first()
    heatmap_key = (gradcam_result.result_metadata or {}).get("heatmap_key") if gradcam_result else None
    if heatmap_key is None:
        raise HTTPException(status_code=404, detail="heatmap not available for this scan")

    return Response(content=download_media(heatmap_key), media_type="image/png")
