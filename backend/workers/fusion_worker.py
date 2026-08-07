from sqlalchemy import select

from app.audit import write_audit_log
from app.db import get_session_maker
from app.db_models import Scan, ScanResult, ScanStatus
from app.fusion.scoring import fuse
from workers.base_consumer import StreamConsumer

# Which per-modality detectors must report in before a scan can be fused,
# keyed by media type.
EXPECTED_MODELS = {
    "video": {"frame_classifier", "audio_deepfake", "lipsync"},
    "image": {"frame_classifier"},
}


class FusionWorker(StreamConsumer):
    """Consumes completed per-model outputs for a scan and writes the fused verdict."""

    stream_name = "scan.model_outputs"
    consumer_group = "fusion"

    async def handle(self, message: dict) -> None:
        scan_id = message["scan_id"]

        session_maker = get_session_maker()
        async with session_maker() as session:
            scan = await session.get(Scan, scan_id)
            if scan is None:
                return

            result = await session.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))
            results = result.scalars().all()

            expected = EXPECTED_MODELS.get(scan.media_type, {"frame_classifier"})
            seen = {r.model_name for r in results}
            if not expected.issubset(seen):
                scan.status = ScanStatus.PROCESSING.value
                await session.commit()
                return

            model_outputs = {
                r.model_name: {
                    "score": r.score,
                    "confidence": r.confidence,
                    "raw": r.raw,
                    "metadata": r.result_metadata,
                }
                for r in results
            }

            try:
                verdict = fuse(model_outputs)
                scan.fused_score = verdict.get("score")
                scan.fused_verdict = verdict.get("verdict")
                scan.status = ScanStatus.COMPLETED.value
                status = "ok"
            except NotImplementedError:
                # Stacking head not trained/integrated yet — surface this as
                # a distinct, honest status rather than a fake verdict.
                scan.status = ScanStatus.BLOCKED.value
                scan.error = "fusion head not yet integrated"
                status = "blocked"

            await write_audit_log(
                session,
                user_id=None,
                action="fusion.compute",
                resource_id=scan_id,
                metadata={"status": status},
            )
            await session.commit()
