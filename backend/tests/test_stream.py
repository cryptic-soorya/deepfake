import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app
from app.routers import stream as stream_router


@pytest.fixture
def client(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    import asyncio

    asyncio.run(_create_tables(engine))

    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_db():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    monkeypatch.setattr("app.routers.stream.get_session_maker", lambda: session_maker)

    # Models are process-wide singletons now (loaded once and reused across
    # sessions, see stream._ensure_models_loaded), so each test needs a clean
    # load state or an earlier test's monkeypatched load()/predict() would
    # leak into a later one via the cached _video_loaded/_audio_loaded flags.
    monkeypatch.setattr(stream_router, "_video_loaded", False)
    monkeypatch.setattr(stream_router, "_audio_loaded", False)
    monkeypatch.setattr(stream_router, "_video_load_failed", False)
    monkeypatch.setattr(stream_router, "_audio_load_failed", False)

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def test_stream_video_tag_routes_to_frame_classifier(client, monkeypatch):
    monkeypatch.setattr(
        stream_router.EfficientNetB4SBI, "load", lambda self: None
    )
    monkeypatch.setattr(
        stream_router.EfficientNetB4SBI,
        "predict",
        lambda self, payload: {"score": 0.83, "confidence": 0.9},
    )
    monkeypatch.setattr(
        stream_router.AASISTVoiceDetector, "load", lambda self: (_ for _ in ()).throw(NotImplementedError)
    )

    with client.websocket_connect("/api/v1/stream/ws/test-session") as ws:
        ws.send_bytes(bytes([stream_router.TAG_VIDEO]) + b"fake-jpeg-bytes")
        msg = ws.receive_json()

    assert msg == {
        "session_id": "test-session",
        "modality": "video",
        "status": "ok",
        "score": 0.83,
        "confidence": 0.9,
    }


def test_stream_audio_tag_routes_to_voice_detector_and_inverts_score(client, monkeypatch):
    monkeypatch.setattr(
        stream_router.EfficientNetB4SBI, "load", lambda self: (_ for _ in ()).throw(NotImplementedError)
    )
    monkeypatch.setattr(stream_router.AASISTVoiceDetector, "load", lambda self: None)
    monkeypatch.setattr(
        stream_router.AASISTVoiceDetector,
        "predict",
        lambda self, payload: {"score": 0.9, "confidence": 0.95},  # P(real) = 0.9
    )

    with client.websocket_connect("/api/v1/stream/ws/test-session") as ws:
        ws.send_bytes(bytes([stream_router.TAG_AUDIO]) + b"fake-wav-bytes")
        msg = ws.receive_json()

    assert msg["modality"] == "audio"
    assert msg["status"] == "ok"
    assert msg["score"] == pytest.approx(0.1)  # inverted onto the P(fake) axis


def test_stream_silent_audio_chunk_reports_no_speech_not_fake(client, monkeypatch):
    """Background noise / no speech must not be scored as a spoofed voice — the
    detector wraps predict() to return a null score with metadata.silence=True
    for near-silent chunks (see app/models/audio_deepfake.py's RMS gate)."""
    monkeypatch.setattr(
        stream_router.EfficientNetB4SBI, "load", lambda self: (_ for _ in ()).throw(NotImplementedError)
    )
    monkeypatch.setattr(stream_router.AASISTVoiceDetector, "load", lambda self: None)
    monkeypatch.setattr(
        stream_router.AASISTVoiceDetector,
        "predict",
        lambda self, payload: {
            "score": None,
            "confidence": None,
            "raw": None,
            "metadata": {"silence": True, "rms": 0.001},
        },
    )

    with client.websocket_connect("/api/v1/stream/ws/test-session") as ws:
        ws.send_bytes(bytes([stream_router.TAG_AUDIO]) + b"silent-chunk")
        msg = ws.receive_json()

    assert msg["modality"] == "audio"
    assert msg["status"] == "no_speech"
    assert msg["score"] is None


def test_stream_blocked_model_reports_blocked_status(client, monkeypatch):
    monkeypatch.setattr(
        stream_router.EfficientNetB4SBI, "load", lambda self: (_ for _ in ()).throw(NotImplementedError)
    )
    monkeypatch.setattr(
        stream_router.AASISTVoiceDetector, "load", lambda self: (_ for _ in ()).throw(NotImplementedError)
    )

    with client.websocket_connect("/api/v1/stream/ws/test-session") as ws:
        ws.send_bytes(bytes([stream_router.TAG_VIDEO]) + b"fake-jpeg-bytes")
        msg = ws.receive_json()

    assert msg["status"] == "blocked_on_model_integration"
    assert msg["score"] is None


def test_stream_load_failure_other_than_not_implemented_degrades_to_blocked(client, monkeypatch):
    """A real load() failure (bad checkpoint, missing dep, network error, ...) must not
    take down the whole socket — it should degrade to blocked_on_model_integration just
    like the NotImplementedError stub case."""
    monkeypatch.setattr(
        stream_router.EfficientNetB4SBI, "load", lambda self: (_ for _ in ()).throw(RuntimeError("checkpoint fetch failed"))
    )
    monkeypatch.setattr(
        stream_router.AASISTVoiceDetector, "load", lambda self: (_ for _ in ()).throw(ModuleNotFoundError("no gdown"))
    )

    with client.websocket_connect("/api/v1/stream/ws/test-session") as ws:
        ws.send_bytes(bytes([stream_router.TAG_VIDEO]) + b"fake-jpeg-bytes")
        video_msg = ws.receive_json()
        ws.send_bytes(bytes([stream_router.TAG_AUDIO]) + b"fake-wav-bytes")
        audio_msg = ws.receive_json()

    assert video_msg["status"] == "blocked_on_model_integration"
    assert video_msg["score"] is None
    assert audio_msg["status"] == "blocked_on_model_integration"
    assert audio_msg["score"] is None


def test_stream_predict_failure_reports_blocked_but_keeps_session_alive(client, monkeypatch):
    """A transient predict() failure (e.g. a corrupt single frame) should report
    blocked_on_model_integration for that message only, without permanently disabling
    the model or killing the socket — the next good frame should still score."""
    monkeypatch.setattr(stream_router.EfficientNetB4SBI, "load", lambda self: None)
    monkeypatch.setattr(
        stream_router.AASISTVoiceDetector, "load", lambda self: (_ for _ in ()).throw(NotImplementedError)
    )

    predict_calls = {"count": 0}

    def flaky_predict(self, payload):
        predict_calls["count"] += 1
        if predict_calls["count"] == 1:
            raise ValueError("corrupt frame")
        return {"score": 0.42, "confidence": 0.8}

    monkeypatch.setattr(stream_router.EfficientNetB4SBI, "predict", flaky_predict)

    with client.websocket_connect("/api/v1/stream/ws/test-session") as ws:
        ws.send_bytes(bytes([stream_router.TAG_VIDEO]) + b"bad-frame")
        first = ws.receive_json()
        ws.send_bytes(bytes([stream_router.TAG_VIDEO]) + b"good-frame")
        second = ws.receive_json()

    assert first["status"] == "blocked_on_model_integration"
    assert first["score"] is None
    assert second == {
        "session_id": "test-session",
        "modality": "video",
        "status": "ok",
        "score": 0.42,
        "confidence": 0.8,
    }
