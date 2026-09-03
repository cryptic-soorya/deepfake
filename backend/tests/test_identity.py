import io

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app

EMBEDDING_A = [1.0] + [0.0] * 511


class FakeEmbedder:
    def __init__(self, embedding, det_score=0.99):
        self.embedding = embedding
        self.det_score = det_score

    def predict(self, image_bytes):
        return {
            "score": self.det_score,
            "confidence": self.det_score,
            "raw": {"faces": [{"bbox": [0, 0, 1, 1], "det_score": self.det_score, "embedding": self.embedding}]},
            "metadata": {"mode": "enroll", "num_faces": 1, "embedding_dim": 512},
        }


class FakeLiveness:
    def __init__(self, is_live=True, score=0.9):
        self.is_live = is_live
        self.score = score

    def predict(self, image_bytes):
        return {
            "score": self.score,
            "confidence": self.score,
            "raw": {"prediction": [[0, self.score, 0]], "label": 1},
            "metadata": {"is_live": self.is_live, "bbox": [0, 0, 1, 1]},
        }


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

    fake_embedder = FakeEmbedder(EMBEDDING_A)
    fake_liveness = FakeLiveness(is_live=True)

    async def _get_embedder():
        return fake_embedder

    async def _get_liveness():
        return fake_liveness

    monkeypatch.setattr("app.routers.identity._get_embedder", _get_embedder)
    monkeypatch.setattr("app.routers.identity._get_liveness", _get_liveness)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


async def test_enroll_then_verify_matches(client):
    enroll_response = await client.post(
        "/api/v1/identity/enroll",
        data={"user_id": "alice"},
        files={"file": ("face.jpg", io.BytesIO(b"fake image bytes"), "image/jpeg")},
    )
    assert enroll_response.status_code == 200
    assert enroll_response.json()["enrolled"] is True

    verify_response = await client.post(
        "/api/v1/identity/verify",
        data={"user_id": "alice"},
        files={"file": ("face.jpg", io.BytesIO(b"fake image bytes"), "image/jpeg")},
    )
    assert verify_response.status_code == 200
    body = verify_response.json()
    assert body["is_match"] is True
    assert body["liveness"]["is_live"] is True
    assert body["verified"] is True


async def test_verify_unknown_user_returns_404(client):
    response = await client.post(
        "/api/v1/identity/verify",
        data={"user_id": "nobody"},
        files={"file": ("face.jpg", io.BytesIO(b"fake image bytes"), "image/jpeg")},
    )
    assert response.status_code == 404
