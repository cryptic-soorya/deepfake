import tempfile
from pathlib import Path

from app.audit import write_audit_log
from app.db import get_session_maker
from app.db_models import ScanResult
from app.models.frame_classifier import EfficientNetB4SBI
from app.pipeline.frame_sampler import sample_frames
from app.queue import enqueue
from app.storage import download_media
from workers.base_consumer import StreamConsumer

SAMPLE_FPS = 2.0
# Bounds per-scan compute on a laptop-class inference worker; revisit once
# the GRU/temporal-attention head (PLAN.md) makes per-frame cost matter less.
MAX_FRAMES_PER_SCAN = 30


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
        media_type = message.get("media_type", "image")

        if not self._model_loaded:
            try:
                self.model.load()
                self._model_loaded = True
            except NotImplementedError:
                pass

        try:
            media_bytes = download_media(media_key)
            if media_type == "video":
                output = self._predict_video(media_bytes, media_key)
            else:
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

    def _predict_video(self, media_bytes: bytes, media_key: str) -> dict:
        """Runs the (frame-level, not temporal) classifier over frames sampled
        at SAMPLE_FPS and averages the per-frame fake-probability/confidence.

        This is an interim aggregation until the GRU/temporal-attention head
        (PLAN.md) lands — a mean score misses motion artifacts (unnatural
        blinking, frame-blend jitter) that a temporal model would catch.
        """
        suffix = Path(media_key).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(media_bytes)
            tmp.flush()

            scores: list[float] = []
            confidences: list[float] = []
            frames_with_no_face = 0
            for i, frame in enumerate(sample_frames(tmp.name, fps=SAMPLE_FPS)):
                if i >= MAX_FRAMES_PER_SCAN:
                    break
                result = self.model.predict(frame)
                if result["score"] is None:
                    frames_with_no_face += 1
                    continue
                scores.append(result["score"])
                confidences.append(result["confidence"])

        if not scores:
            return {
                "score": None,
                "confidence": None,
                "raw": None,
                "metadata": {
                    "error": "no face detected in any sampled frame",
                    "frames_with_no_face": frames_with_no_face,
                },
            }

        return {
            "score": sum(scores) / len(scores),
            "confidence": sum(confidences) / len(confidences),
            "raw": {"per_frame_scores": scores},
            "metadata": {
                "num_frames_scored": len(scores),
                "frames_with_no_face": frames_with_no_face,
                "aggregation": "mean_frame_score_interim_pending_temporal_head",
                "interpretation": "score is mean P(fake) across sampled frames; high score indicates a deepfake video",
            },
        }
