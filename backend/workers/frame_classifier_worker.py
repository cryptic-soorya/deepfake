from app.audit import write_audit_log
from app.db import get_session_maker
from app.db_models import ScanResult
from app.models.frame_classifier import EfficientNetB4SBI
from app.queue import enqueue
from app.storage import download_media
from workers.base_consumer import StreamConsumer


class FrameClassifierWorker(StreamConsumer):
    stream_name = "scan.frames"
    consumer_group = "frame-detector"
    model_name = "frame_classifier"

    def __init__(self) -> None:
        self.model = EfficientNetB4SBI()
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
            # Weights for this detector haven't been trained/integrated yet
            # (per CLAUDE.md, EfficientNet-B4/SBI fine-tuning happens on a
            # rented GPU, not locally). Record the gap honestly rather than
            # fabricating a score.
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
