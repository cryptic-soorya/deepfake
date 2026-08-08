from app.audit import write_audit_log
from app.db import get_session_maker
from app.db_models import ScanResult
from app.models.face_detector import SCRFDFaceDetector
from app.models.lipsync import SyncNetLipSync
from app.pipeline.audio_extract import extract_audio
from app.pipeline.face_track import track_faces
from app.queue import enqueue
from app.storage import download_media
from workers.base_consumer import StreamConsumer

# Must match SyncNetLipSync's expected input convention (25fps, 224x224
# mouth/face-centered crops) -- see app/models/lipsync.py.
VIDEO_FPS = 25
CROP_SIZE = 224
AUDIO_SAMPLE_RATE = 16000


class LipsyncWorker(StreamConsumer):
    stream_name = "scan.lipsync"
    consumer_group = "lipsync-detector"
    model_name = "lipsync"

    def __init__(self) -> None:
        self.model = SyncNetLipSync()
        self.detector = SCRFDFaceDetector()
        self._model_loaded = False
        self._detector_loaded = False

    async def warmup(self) -> None:
        if not self._detector_loaded:
            try:
                self.detector.load()
                self._detector_loaded = True
            except NotImplementedError:
                pass

        if not self._model_loaded:
            try:
                self.model.load()
                self._model_loaded = True
            except NotImplementedError:
                pass

    async def handle(self, message: dict) -> None:
        scan_id = message["scan_id"]
        media_key = message["media_key"]

        if not self._detector_loaded:
            try:
                self.detector.load()
                self._detector_loaded = True
            except NotImplementedError:
                pass

        if not self._model_loaded:
            try:
                self.model.load()
                self._model_loaded = True
            except NotImplementedError:
                pass

        try:
            media_bytes = download_media(media_key)
            track = track_faces(
                media_bytes,
                target_fps=VIDEO_FPS,
                crop_size=CROP_SIZE,
                detector=self.detector if self._detector_loaded else None,
            )

            if track["num_frames_with_face"] == 0:
                output = {
                    "score": None,
                    "confidence": None,
                    "raw": None,
                    "metadata": {"status": "no_face_tracked_in_video"},
                }
            else:
                waveform = extract_audio(media_bytes, target_sr=AUDIO_SAMPLE_RATE)
                if waveform.size == 0:
                    output = {
                        "score": None,
                        "confidence": None,
                        "raw": None,
                        "metadata": {"status": "no_audio_track_decoded"},
                    }
                else:
                    output = self.model.predict({"lip_frames": track["frames"], "audio": waveform})
                    output.setdefault("metadata", {})
                    output["metadata"]["num_frames_tracked"] = track["num_frames"]
                    output["metadata"]["held_frames"] = track["held_frames"]
            status = "ok"
        except NotImplementedError:
            # SyncNet checkpoint not integrated yet -- record the gap
            # honestly instead of fabricating a score (see CLAUDE.md).
            output = {"score": None, "confidence": None, "raw": None, "metadata": {"status": "model_not_integrated"}}
            status = "blocked"

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
            await write_audit_log(
                session,
                user_id=None,
                action=f"model.{self.model_name}.predict",
                resource_id=scan_id,
                metadata={"status": status},
            )
            await session.commit()

        await enqueue("scan.model_outputs", {"scan_id": scan_id, "model_name": self.model_name, "status": status})
