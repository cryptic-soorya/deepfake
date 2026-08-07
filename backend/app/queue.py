"""Redis Streams producer helper — the inference pipeline's message queue.

Per CLAUDE.md: Redis Streams first, Kafka only if load-testing proves it
insufficient. Don't reach for Kafka by default.
"""
import json
from functools import lru_cache

import redis.asyncio as redis

from app.config import get_settings


@lru_cache
def get_redis() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


def _encode(value: object) -> str | int | float:
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value)


async def enqueue(stream: str, fields: dict) -> str:
    """Add a message to a Redis stream. Field values are JSON-encoded unless already scalar."""
    r = get_redis()
    encoded = {k: _encode(v) for k, v in fields.items()}
    return await r.xadd(stream, encoded)
