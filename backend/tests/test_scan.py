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

    enqueued: list[tuple[str, dict]] = []

    async def fake_enqueue(stream, fields):
        enqueued.append((stream, fields))
        return "0-1"

    monkeypatch.setattr("app.routers.scan.enqueue", fake_enqueue)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.enqueued = enqueued
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


async def test_create_scan_uploads_and_enqueues(client):
    response = await client.post(
        "/api/v1/scan/",
        files={"file": ("clip.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["scan_id"]

    stream_names = {stream for stream, _ in client.enqueued}
    assert stream_names == {"scan.frames", "scan.audio"}


async def test_create_scan_image_only_enqueues_frames(client):
    response = await client.post(
        "/api/v1/scan/",
        files={"file": ("photo.png", io.BytesIO(b"fake image bytes"), "image/png")},
    )
    assert response.status_code == 200

    stream_names = {stream for stream, _ in client.enqueued}
    assert stream_names == {"scan.frames"}


async def test_get_scan_returns_pending_status(client):
    create_response = await client.post(
        "/api/v1/scan/",
        files={"file": ("clip.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")},
    )
    scan_id = create_response.json()["scan_id"]

    get_response = await client.get(f"/api/v1/scan/{scan_id}")
    assert get_response.status_code == 200
    detail = get_response.json()
    assert detail["scan_id"] == scan_id
    assert detail["media_type"] == "video"
    assert detail["status"] == "pending"
    assert detail["model_results"] == []


async def test_get_missing_scan_returns_404(client):
    response = await client.get("/api/v1/scan/does-not-exist")
    assert response.status_code == 404
