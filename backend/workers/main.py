"""Entrypoint that runs every queue consumer as a concurrent task.

Usage: python -m workers.main
"""
import asyncio
import logging

from workers.audio_worker import AudioWorker
from workers.frame_classifier_worker import FrameClassifierWorker
from workers.fusion_worker import FusionWorker
from workers.lipsync_worker import LipsyncWorker

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    workers = [FrameClassifierWorker(), AudioWorker(), LipsyncWorker(), FusionWorker()]
    await asyncio.gather(*(worker.run() for worker in workers))


if __name__ == "__main__":
    asyncio.run(main())
