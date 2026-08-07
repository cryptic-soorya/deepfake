from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def stream_session(websocket: WebSocket, session_id: str):
    """Live webcam session: receive frames/audio chunks, emit rolling verdicts."""
    await websocket.accept()
    raise NotImplementedError
