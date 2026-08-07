from app.models.audio_deepfake import AASISTVoiceDetector
from workers.base_consumer import StreamConsumer


class AudioWorker(StreamConsumer):
    stream_name = "scan.audio"
    consumer_group = "audio-detector"

    def __init__(self) -> None:
        self.model = AASISTVoiceDetector()

    async def handle(self, message: dict) -> None:
        raise NotImplementedError
