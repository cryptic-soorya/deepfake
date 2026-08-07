"""Shared Redis Streams consumer base — each model worker subclasses this."""


class StreamConsumer:
    stream_name: str
    consumer_group: str

    async def handle(self, message: dict) -> None:
        raise NotImplementedError

    async def run(self) -> None:
        raise NotImplementedError
