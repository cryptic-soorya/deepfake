from app.models.frame_classifier import EfficientNetB4SBI
from workers.base_consumer import StreamConsumer


class FrameClassifierWorker(StreamConsumer):
    stream_name = "scan.frames"
    consumer_group = "frame-classifier"

    def __init__(self) -> None:
        self.model = EfficientNetB4SBI()

    async def handle(self, message: dict) -> None:
        raise NotImplementedError
