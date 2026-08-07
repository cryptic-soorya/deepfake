from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.db import get_db
from app.db_models import Scan
from app.reports.generate import generate_report

router = APIRouter()


@router.get("/{scan_id}")
async def get_report(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Return the signed forensic report (PDF/JSON) for a scan."""
    scan = await db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    try:
        report_bytes = generate_report(scan_id)
        status = "ok"
    except NotImplementedError:
        status = "blocked_on_model_integration"

    await write_audit_log(
        db,
        user_id=None,
        action="report.generate",
        resource_id=scan_id,
        metadata={"status": status},
    )
    await db.commit()

    if status != "ok":
        raise HTTPException(status_code=501, detail="report generation not yet integrated")

    return Response(content=report_bytes, media_type="application/pdf")
