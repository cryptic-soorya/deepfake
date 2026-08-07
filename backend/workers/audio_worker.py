from app.audit import write_audit_log
from app.db import get_session_maker
from app.db_models import ScanResult
from app.models.audio_deepfake import AASISTVoiceDetector
from app.queue import enqueue
from app.storage import download_media
from workers.base_consumer import StreamConsumer


class AudioWorker(StreamConsumer):
    stream_name = "scan.audio"
    consumer_group = "audio-detector"
    model_name = "audio_deepfake"

    def __init__(self) -> None:
        self.model = AASISTVoiceDetector()
        self._model_loaded = False

    async def handle(self, message: dict) -> None:
        scan_id = message["scan_id"]
        media_key = message["media_key"]

        if not self._model_loaded:
            try:
                self.model.load()
                self._model_loaded = True
            except NotImplementedError:
                pass

        try:
            media_bytes = download_media(media_key)
            output = self.model.predict(media_bytes)
            status = "ok"
        except NotImplementedError:
            # AASIST + XLSR checkpoint not integrated yet — record the gap
            # honestly instead of fabricating a score (see CLAUDE.md).
            output = {"score": None, "confidence": None, "raw": None, "metadata": {"status": "model_not_integrated"}}
            status = "blocked"

        session_maker = get_session_maker()
        async with session_maker() as session:
            session.add(
                ScanResult(
                    scan_id=scan_id,
                    model_name=self.model_name,
                    score=output.get("score"),
                    confidence=output.get("confidence"),
                    raw=output.get("raw"),
                    result_metadata=output.get("metadata"),
                )
            )
            await write_audit_log(
                session,
                user_id=None,
                action=f"model.{self.model_name}.predict",
                resource_id=scan_id,
                metadata={"status": status},
            )
            await session.commit()

        await enqueue("scan.model_outputs", {"scan_id": scan_id, "model_name": self.model_name, "status": status})
