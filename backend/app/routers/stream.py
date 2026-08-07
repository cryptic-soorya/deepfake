from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.audit import write_audit_log
from app.db import get_session_maker
from app.models.frame_classifier import EfficientNetB4SBI

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def stream_session(websocket: WebSocket, session_id: str):
    """Live webcam session: receive frames/audio chunks, emit rolling verdicts.

    Frame-level classifier isn't trained/integrated yet (see EfficientNetB4SBI),
    so each frame currently gets an honest "blocked" verdict instead of a
    fabricated score, same pattern the batch scan workers use.
    """
    await websocket.accept()

    session_maker = get_session_maker()
    async with session_maker() as session:
        await write_audit_log(
            session, user_id=None, action="stream.session_start", resource_id=session_id, metadata=None
        )
        await session.commit()

    model = EfficientNetB4SBI()
    model_loaded = False
    try:
        model.load()
        model_loaded = True
    except NotImplementedError:
        pass

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()

            if model_loaded:
                try:
                    output = model.predict(frame_bytes)
                    await websocket.send_json(
                        {
                            "session_id": session_id,
                            "status": "ok",
                            "score": output.get("score"),
                            "confidence": output.get("confidence"),
                        }
                    )
                    continue
                except NotImplementedError:
                    model_loaded = False

            await websocket.send_json(
                {
                    "session_id": session_id,
                    "status": "blocked_on_model_integration",
                    "score": None,
                    "confidence": None,
                }
            )
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
