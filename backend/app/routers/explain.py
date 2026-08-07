from fastapi import APIRouter

router = APIRouter()


@router.get("/{scan_id}")
async def explain_scan(scan_id: str):
    """Return the Claude-generated narrative explanation for a scan verdict."""
    raise NotImplementedError
