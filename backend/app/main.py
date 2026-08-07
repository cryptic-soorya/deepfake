from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import scan, identity, stream, explain, report

app = FastAPI(title="Morpheus.AI", version="0.1.0")

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


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
