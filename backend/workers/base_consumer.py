"""Shared Redis Streams consumer base — each model worker subclasses this."""
import logging

from app.queue import get_redis

logger = logging.getLogger(__name__)


class StreamConsumer:
    stream_name: str
    consumer_group: str
    consumer_name: str = "worker-1"
    block_ms: int = 5000

    async def handle(self, message: dict) -> None:
        raise NotImplementedError

    async def run(self) -> None:
        redis = get_redis()
        try:
            await redis.xgroup_create(self.stream_name, self.consumer_group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

        logger.info("worker started stream=%s group=%s", self.stream_name, self.consumer_group)
        while True:
            response = await redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: ">"},
                count=10,
                block=self.block_ms,
            )
            if not response:
                continue

            for _stream, messages in response:
                for message_id, fields in messages:
                    try:
                        await self.handle(fields)
                    except Exception:
                        logger.exception(
                            "handler failed stream=%s message_id=%s", self.stream_name, message_id
                        )
                        continue
                    await redis.xack(self.stream_name, self.consumer_group, message_id)
