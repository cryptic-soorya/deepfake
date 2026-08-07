import io

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app


@pytest.fixture
async def client(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_db():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    monkeypatch.setattr("app.routers.scan.upload_media", lambda *a, **k: None)

    async def fake_enqueue(stream, fields):
        return "0-1"

    monkeypatch.setattr("app.routers.scan.enqueue", fake_enqueue)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


async def _create_scan(client) -> str:
    response = await client.post(
        "/api/v1/scan/",
        files={"file": ("clip.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")},
    )
    return response.json()["scan_id"]


async def test_explain_returns_501_when_model_not_integrated(client):
    scan_id = await _create_scan(client)
    response = await client.get(f"/api/v1/explain/{scan_id}")
    assert response.status_code == 501


async def test_explain_missing_scan_returns_404(client):
    response = await client.get("/api/v1/explain/does-not-exist")
    assert response.status_code == 404


async def test_report_returns_501_when_generation_not_integrated(client):
    scan_id = await _create_scan(client)
    response = await client.get(f"/api/v1/report/{scan_id}")
    assert response.status_code == 501


async def test_report_missing_scan_returns_404(client):
    response = await client.get("/api/v1/report/does-not-exist")
    assert response.status_code == 404
