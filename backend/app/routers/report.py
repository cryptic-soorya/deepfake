from fastapi import APIRouter

router = APIRouter()


@router.get("/{scan_id}")
async def get_report(scan_id: str):
    """Return the signed forensic report (PDF/JSON) for a scan."""
    raise NotImplementedError
