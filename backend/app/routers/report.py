from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.db import get_db
from app.db_models import Scan, ScanResult
from app.reports.generate import generate_report

router = APIRouter()


@router.get("/{scan_id}")
async def get_report(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Return the signed forensic report (PDF) for a completed scan."""
    scan = await db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    if scan.status not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail="scan still in progress — report not ready yet")

    results = (await db.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))).scalars().all()
    report_bytes = generate_report(scan, results)

    await write_audit_log(
        db,
        user_id=None,
        action="report.generate",
        resource_id=scan_id,
        metadata={"status": "ok"},
    )
    await db.commit()

    headers = {"Content-Disposition": f'attachment; filename="morpheus-report-{scan_id}.pdf"'}
    return Response(content=report_bytes, media_type="application/pdf", headers=headers)
