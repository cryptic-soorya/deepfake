import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.audit import write_audit_log
from app.db import get_session_maker
from app.models.audio_deepfake import AASISTVoiceDetector
from app.models.frame_classifier import EfficientNetB4SBI

router = APIRouter()
logger = logging.getLogger(__name__)

# Each binary WS message is prefixed with a 1-byte modality tag so a single
# socket can carry both the ~2fps JPEG frame stream and the ~4s audio-chunk
# stream without a second connection.
TAG_VIDEO = 0x01
TAG_AUDIO = 0x02

# Models are process-wide singletons, loaded at most once and reused by every
# session. Loading these (SCRFD + EfficientNet-B4 for video, AASIST + XLSR
# for audio, including a first-run checkpoint download) takes many seconds --
# doing it per-connection, as this used to, meant every single session sat on
# "no signal" for that long before its first real verdict. `_load_lock`
# serializes the one-time load so two sessions connecting at once don't race
# and load twice.
_video_model = EfficientNetB4SBI()
_audio_model = AASISTVoiceDetector()
_video_loaded = False
_audio_loaded = False
_video_load_failed = False
_audio_load_failed = False
_load_lock = asyncio.Lock()


async def _ensure_models_loaded() -> tuple[bool, bool]:
    global _video_loaded, _audio_loaded, _video_load_failed, _audio_load_failed

    if (_video_loaded or _video_load_failed) and (_audio_loaded or _audio_load_failed):
        return _video_loaded, _audio_loaded

    async with _load_lock:
        if not _video_loaded and not _video_load_failed:
            try:
                await asyncio.to_thread(_video_model.load)
                _video_loaded = True
            except Exception:
                logger.exception("video model failed to load; streaming will report blocked_on_model_integration")
                _video_load_failed = True

        if not _audio_loaded and not _audio_load_failed:
            try:
                await asyncio.to_thread(_audio_model.load)
                _audio_loaded = True
            except Exception:
                logger.exception("audio model failed to load; streaming will report blocked_on_model_integration")
                _audio_load_failed = True

    return _video_loaded, _audio_loaded


@router.websocket("/ws/{session_id}")
async def stream_session(websocket: WebSocket, session_id: str):
    """Live webcam session: receive frames/audio chunks, emit rolling verdicts.

    Video frames go to the frame-level classifier; audio chunks go to the
    AASIST voice-clone detector. AASIST's native score is P(real/bonafide)
    (see app/models/audio_deepfake.py), inverted here onto the same P(fake)
    axis the video score uses, mirroring app/fusion/scoring.py's
    MODALITY_DIRECTION convention so both channels read the same way on the
    client.
    """
    await websocket.accept()

    session_maker = get_session_maker()
    async with session_maker() as session:
        await write_audit_log(
            session, user_id=None, action="stream.session_start", resource_id=session_id, metadata=None
        )
        await session.commit()

    video_loaded, audio_loaded = await _ensure_models_loaded()

    try:
        while True:
            try:
                raw = await websocket.receive_bytes()
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("failed to receive a message on session %s; waiting for the next one", session_id)
                continue

            if len(raw) < 2:
                continue
            tag, payload = raw[0], raw[1:]
            modality = "audio" if tag == TAG_AUDIO else "video"

            score = None
            confidence = None
            status = "blocked_on_model_integration"

            if modality == "audio" and audio_loaded:
                try:
                    output = await asyncio.to_thread(_audio_model.predict, payload)
                    raw_score = output.get("score")
                    if raw_score is None and (output.get("metadata") or {}).get("silence"):
                        status = "no_speech"
                    else:
                        score = (1.0 - raw_score) if raw_score is not None else None
                        confidence = output.get("confidence")
                        status = "ok"
                except NotImplementedError:
                    audio_loaded = False
                except Exception:
                    logger.exception("audio prediction failed on session %s; reporting blocked for this chunk", session_id)
            elif modality == "video" and video_loaded:
                try:
                    output = await asyncio.to_thread(_video_model.predict, payload)
                    score = output.get("score")
                    confidence = output.get("confidence")
                    status = "ok"
                except NotImplementedError:
                    video_loaded = False
                except Exception:
                    logger.exception("video prediction failed on session %s; reporting blocked for this frame", session_id)

            try:
                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "modality": modality,
                        "status": status,
                        "score": score,
                        "confidence": confidence,
                    }
                )
            except Exception:
                logger.exception("failed to send result on session %s", session_id)
    except WebSocketDisconnect:
        pass
    finally:
        session_maker = get_session_maker()
        async with session_maker() as session:
            await write_audit_log(
                session, user_id=None, action="stream.session_end", resource_id=session_id, metadata=None
            )
            await session.commit()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
