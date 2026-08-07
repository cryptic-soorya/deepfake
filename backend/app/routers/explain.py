from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.db import get_db
from app.db_models import Scan, ScanResult
from app.models.claude_explainer import ClaudeExplainer

router = APIRouter()


@lru_cache
def _get_explainer() -> ClaudeExplainer:
    model = ClaudeExplainer()
    model.load()
    return model


@router.get("/{scan_id}")
async def explain_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Return the Claude-generated narrative explanation for a scan verdict."""
    scan = await db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    if scan.explanation:
        return {"scan_id": scan_id, "explanation": scan.explanation, "status": "ok"}

    results = (await db.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))).scalars().all()
    model_outputs = {
        r.model_name: {"score": r.score, "confidence": r.confidence, "metadata": r.result_metadata}
        for r in results
    }

    try:
        explainer = _get_explainer()
        explanation = explainer.predict(
            {
                "fused_score": scan.fused_score,
                "fused_verdict": scan.fused_verdict,
                "model_outputs": model_outputs,
            }
        )["metadata"]["narrative"]
        scan.explanation = explanation
        status = "ok"
    except NotImplementedError:
        explanation = None
        status = "blocked_on_model_integration"

    await write_audit_log(
        db,
        user_id=None,
        action="explain.generate",
        resource_id=scan_id,
        metadata={"status": status},
    )
    await db.commit()

    if status != "ok":
        raise HTTPException(status_code=501, detail="Claude explainer not yet integrated")

    return {"scan_id": scan_id, "explanation": explanation, "status": status}
