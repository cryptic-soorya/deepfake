import base64
import io

from app.audit import write_audit_log
from app.db import get_session_maker
from app.db_models import ScanResult
from app.models.frame_classifier import EfficientNetB4SBI
from app.models.gradcam import GradCAMExplainer
from app.pipeline.frame_sampler import sample_frames
from app.queue import enqueue
from app.storage import download_media, upload_media
from workers.base_consumer import StreamConsumer

# Target sampling rate for video scans and the hard cap on frames scored per
# scan -- bounds per-scan inference cost. See app/pipeline/frame_sampler.py
# for why this is fixed-rate rather than scene-change/face-track-aware.
VIDEO_SAMPLE_FPS = 2.0
VIDEO_MAX_FRAMES = 32


def _aggregate_frame_predictions(model, frames: list[dict]) -> dict:
    """Run the classifier on each sampled frame and fold per-frame results into
    the single {score, confidence, raw, metadata} contract the fusion layer
    expects (one ScanResult row per model per scan, not one per frame).

    Aggregation is max-over-frames on P(fake): a single convincingly faked
    frame is evidence of manipulation even if most frames look clean, so
    averaging would dilute exactly the signal we want to catch. Frames where
    no face was found are excluded from aggregation but kept in `raw` for
    the forensic report / Grad-CAM step to reference later.
    """
    per_frame = []
    scored = []  # list of (output, image) pairs for frames where a face was found
    for sampled in frames:
        output = model.predict(sampled["frame"])
        per_frame.append(
            {
                "index": sampled["index"],
                "timestamp_sec": sampled["timestamp_sec"],
                "score": output.get("score"),
                "confidence": output.get("confidence"),
                "metadata": output.get("metadata"),
            }
        )
        if output.get("score") is not None:
            scored.append((output, sampled["frame"]))

    if not scored:
        return {
            "score": None,
            "confidence": None,
            "raw": {"frames": per_frame},
            "metadata": {
                "status": "no_face_in_any_sampled_frame",
                "num_frames_sampled": len(frames),
            },
            "_best_frame_image": None,
        }

    best, best_frame_image = max(scored, key=lambda pair: pair[0]["score"])
    mean_score = sum(o["score"] for o, _ in scored) / len(scored)

    return {
        "score": best["score"],
        "confidence": best["confidence"],
        "raw": {"frames": per_frame, "best_frame_raw": best["raw"]},
        "metadata": {
            "aggregation": "max_over_sampled_frames",
            "num_frames_sampled": len(frames),
            "num_frames_with_face": len(scored),
            "mean_score": mean_score,
            "interpretation": "score is P(fake) of the most-suspicious sampled frame; high score indicates a deepfake",
        },
        "_best_frame_image": best_frame_image,
    }


class FrameClassifierWorker(StreamConsumer):
    stream_name = "scan.frames"
    consumer_group = "frame-detector"
    model_name = "frame_classifier"

    def __init__(self) -> None:
        self.model = EfficientNetB4SBI()
        self.gradcam = GradCAMExplainer()
        self._model_loaded = False
        self._gradcam_loaded = False

    async def warmup(self) -> None:
        if not self._model_loaded:
            try:
                self.model.load()
                self._model_loaded = True
            except NotImplementedError:
                pass

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

        best_image = None
        try:
            media_bytes = download_media(media_key)
            if media_type == "video":
                frames = sample_frames(media_bytes, fps=VIDEO_SAMPLE_FPS, max_frames=VIDEO_MAX_FRAMES)
                if not frames:
                    output = {
                        "score": None,
                        "confidence": None,
                        "raw": None,
                        "metadata": {"status": "no_frames_decoded_from_video"},
                    }
                else:
                    output = _aggregate_frame_predictions(self.model, frames)
                    best_image = output.pop("_best_frame_image", None)
            else:
                output = self.model.predict(media_bytes)
                if output.get("score") is not None:
                    best_image = media_bytes
            status = "ok"
        except NotImplementedError:
            # Weights for this detector haven't been trained/integrated yet
            # (per CLAUDE.md, EfficientNet-B4/SBI fine-tuning happens on a
            # rented GPU, not locally). Record the gap honestly rather than
            # fabricating a score.
            output = {"score": None, "confidence": None, "raw": None, "metadata": {"status": "model_not_integrated"}}
            status = "blocked"

        gradcam_output, gradcam_status = self._run_gradcam(scan_id, best_image)

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
            if gradcam_output is not None:
                session.add(
                    ScanResult(
                        scan_id=scan_id,
                        model_name="gradcam",
                        score=gradcam_output.get("score"),
                        confidence=gradcam_output.get("confidence"),
                        raw=gradcam_output.get("raw"),
                        result_metadata=gradcam_output.get("metadata"),
                    )
                )
            await write_audit_log(
                session,
                user_id=None,
                action=f"model.{self.model_name}.predict",
                resource_id=scan_id,
                metadata={"status": status},
            )
            if gradcam_output is not None:
                await write_audit_log(
                    session,
                    user_id=None,
                    action="model.gradcam.predict",
                    resource_id=scan_id,
                    metadata={"status": gradcam_status},
                )
            await session.commit()

        await enqueue("scan.model_outputs", {"scan_id": scan_id, "model_name": self.model_name, "status": status})

    def _run_gradcam(self, scan_id: str, image) -> tuple[dict | None, str | None]:
        """Runs Grad-CAM on the most-suspicious frame, if there is one, and
        uploads the heatmap PNG to object storage (per CLAUDE.md: no binary
        media in Postgres) rather than persisting the base64 payload straight
        into the ScanResult row.
        """
        if image is None:
            return None, None

        if not self._gradcam_loaded:
            try:
                self.gradcam.load()
                self._gradcam_loaded = True
            except NotImplementedError:
                return (
                    {"score": None, "confidence": None, "raw": None, "metadata": {"status": "model_not_integrated"}},
                    "blocked",
                )

        try:
            result = self.gradcam.predict(image)
        except NotImplementedError:
            return (
                {"score": None, "confidence": None, "raw": None, "metadata": {"status": "model_not_integrated"}},
                "blocked",
            )

        heatmap_b64 = result.get("metadata", {}).pop("heatmap_png_base64", None)
        if heatmap_b64 is not None:
            heatmap_key = f"gradcam/{scan_id}.png"
            upload_media(heatmap_key, io.BytesIO(base64.b64decode(heatmap_b64)), content_type="image/png")
            result["metadata"]["heatmap_key"] = heatmap_key

        return result, "ok"
