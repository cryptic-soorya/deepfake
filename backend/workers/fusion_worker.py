from app.fusion.scoring import fuse
from workers.base_consumer import StreamConsumer


class FusionWorker(StreamConsumer):
    """Consumes completed per-model outputs for a scan and writes the fused verdict."""

    stream_name = "scan.model_outputs"
    consumer_group = "fusion"

    async def handle(self, message: dict) -> None:
        raise NotImplementedError
