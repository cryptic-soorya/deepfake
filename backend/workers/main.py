"""Entrypoint that runs every queue consumer as its own OS process.

Each worker gets its own process (not just an asyncio task sharing one loop)
for two reasons, both observed in practice:
  1. Model inference (ONNX/torch forward passes, video frame loops) is
     synchronous CPU-bound work with no `await` in it. Sharing one event loop
     across workers means one worker's slow `handle()` call starves every
     other worker's Redis heartbeat, which can time out and raise.
  2. asyncio.gather propagates the first exception and cancels every sibling
     task -- so one worker's crash used to kill all four at once, with
     nothing to bring them back. A dead worker then leaves scans stuck at
     "pending" forever with no visible error.

A lightweight supervisor loop respawns any worker process that exits, so a
crash degrades throughput instead of silently halting the whole pipeline.

Usage: python -m workers.main
"""
import logging
import multiprocessing
import time

from workers.audio_worker import AudioWorker
from workers.frame_classifier_worker import FrameClassifierWorker
from workers.fusion_worker import FusionWorker
from workers.lipsync_worker import LipsyncWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORKER_CLASSES = [FrameClassifierWorker, AudioWorker, LipsyncWorker, FusionWorker]
RESPAWN_BACKOFF_SECONDS = 2


def _run_worker(worker_cls: type) -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(worker_cls().run())


def main() -> None:
    processes: dict[str, multiprocessing.Process] = {}

    def spawn(worker_cls: type) -> None:
        proc = multiprocessing.Process(target=_run_worker, args=(worker_cls,), name=worker_cls.__name__, daemon=True)
        proc.start()
        processes[worker_cls.__name__] = proc
        logger.info("spawned worker=%s pid=%s", worker_cls.__name__, proc.pid)

    for worker_cls in WORKER_CLASSES:
        spawn(worker_cls)

    try:
        while True:
            for worker_cls in WORKER_CLASSES:
                proc = processes[worker_cls.__name__]
                if not proc.is_alive():
                    logger.error("worker=%s pid=%s died (exitcode=%s) -- respawning", proc.name, proc.pid, proc.exitcode)
                    time.sleep(RESPAWN_BACKOFF_SECONDS)
                    spawn(worker_cls)
            time.sleep(1)
    except KeyboardInterrupt:
        for proc in processes.values():
            proc.terminate()


if __name__ == "__main__":
    main()
