import asyncio
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import scan, identity, stream, explain, report

logger = logging.getLogger(__name__)

app = FastAPI(title="Morpheus.AI", version="0.1.0")
app.state.models_ready = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router, prefix="/api/v1/scan", tags=["scan"])
app.include_router(identity.router, prefix="/api/v1/identity", tags=["identity"])
app.include_router(stream.router, prefix="/api/v1/stream", tags=["stream"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["explain"])
app.include_router(report.router, prefix="/api/v1/report", tags=["report"])


@app.on_event("startup")
async def warm_models() -> None:
    """Load every API-process model once at boot instead of on first use.

    Without this, the cold-start tax (ONNX/torch init, first-run checkpoint
    resolution) lands on whichever request happens to hit a given model
    first -- which in practice is a user clicking "Begin Enrollment" or
    opening a live session mid-demo. Loading them concurrently here means
    the wait happens once, during `uvicorn` startup, and the wall-clock
    cost is the slowest single model rather than the sum of all of them.
    Failures are logged, not raised, so a missing GPU/API key locally
    doesn't block the whole server from starting -- the affected endpoint
    just reports blocked_on_model_integration on first real use, as before.
    """
    warmers = [
        ("identity.embedder", identity._get_embedder()),
        ("identity.liveness", identity._get_liveness()),
        ("stream.video+audio", stream._ensure_models_loaded()),
    ]
    started = time.monotonic()
    logger.info("warming %d model(s) before accepting real traffic...", len(warmers))
    results = await asyncio.gather(*(coro for _, coro in warmers), return_exceptions=True)
    for (name, _), result in zip(warmers, results):
        if isinstance(result, Exception):
            logger.warning("model warmup failed for %s: %s", name, result)
    app.state.models_ready = True
    logger.info("model warmup done in %.1fs", time.monotonic() - started)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "models_ready": app.state.models_ready}
