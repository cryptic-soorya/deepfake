# PROGRESS.md — Morpheus.AI Build Log

Running changelog of what's been built, by whom, and what's still stubbed. See `PLAN.md` for the phased
target and `CLAUDE.md` for the locked architecture decisions this work follows. Newest entries first.

Convention: entries note **what changed**, **why**, and honestly flag anything still faked/stubbed —
per CLAUDE.md, no silent placeholders pretending to be real results.

---

## 2026-08-07 — SCRFD face detector wired to real weights

**Who:** cryptic-soorya (via Claude Code)
**Round:** 2 (Prototype)

- `backend/app/models/face_detector.py`: `SCRFDFaceDetector` now loads real InsightFace weights
  (`buffalo_l` pack, `detection` module only — matches the pack CLAUDE.md specifies for ArcFace, so a
  later identity-verification wrapper can reuse the same download) via `insightface.app.FaceAnalysis`,
  and runs genuine ONNX inference in `predict()`. Device mapping goes through a new `_onnx_providers()`
  helper: `mps` → CoreML → CPU, `cuda` → CUDA → CPU, since onnxruntime has no MPS provider and blindly
  reusing the torch device string would have been wrong on the team's M4 dev machines.
- `backend/tests/test_face_detector.py`: integration test that actually loads weights (using the
  lighter `buffalo_sc` pack to keep the test download small) and runs inference — against a synthetic
  blank image (proves the well-formed-empty-result path) and against InsightFace's own bundled sample
  photo `insightface/data/images/t1.jpg` (proves genuine detection: 6/6 faces found, top confidence
  0.88, real bboxes + 5-point landmarks). Skips (not fails) if offline, so CI/offline dev isn't blocked
  on a model download.
- Verified locally: `pytest -v` → 9 passed (4 SCRFD + 1 health + 4 scan pipeline). SCRFD weight download
  (`buffalo_sc`, ~16MB) and first cold-start inference took ~5 minutes on the team's M4; cached under
  `~/.insightface/models` after that.
- **Status:** real, verified — this is the first model in the stack running genuine inference through
  the `ModelWrapper` contract, proving the interface works end-to-end before the heavier models
  (EfficientNet-B4/SBI, AASIST) are integrated.
- **Not done yet:** SCRFD isn't wired into `pipeline/face_track.py` or the scan worker pipeline yet —
  it's proven standalone. Frame-level face tracking across a video is still `NotImplementedError`.

## 2026-08-07 — Scan pipeline plumbing (upload → queue → worker → fusion)

**Who:** cryptic-soorya (via Claude Code)
**Round:** 2 (Prototype)

- `backend/app/db.py`, `backend/app/db_models.py`: async SQLAlchemy models for `Scan`, `ScanResult`,
  `AuditLog`. `backend/migrations/`: Alembic scaffolding + initial migration (`0001_initial_scan_tables`).
- `backend/app/queue.py`: Redis Streams producer helper (`enqueue`). `backend/app/storage.py`:
  MinIO/S3 upload/download via boto3.
- `backend/app/audit.py`: `write_audit_log` now actually writes a row (was `NotImplementedError`) —
  takes the caller's `AsyncSession` so it shares a transaction with the write it's auditing, per
  CLAUDE.md's "audit log in the same PR, same transaction" convention.
- `backend/app/routers/scan.py`: `POST /api/v1/scan/` uploads to object storage, creates a `Scan` row,
  writes an audit log entry, and enqueues to `scan.frames` (+ `scan.audio` if the upload is a video).
  `GET /api/v1/scan/{id}` returns live status + per-model results from Postgres.
- `backend/workers/base_consumer.py`: real Redis Streams consumer-group loop (`XGROUP CREATE` /
  `XREADGROUP` / `XACK`), replacing the stub.
- `backend/workers/frame_classifier_worker.py`, `audio_worker.py`, `fusion_worker.py`: download media,
  call the real `ModelWrapper.predict()` / `fuse()` interfaces. **Honesty note:** the actual detectors
  (EfficientNet-B4/SBI, AASIST+XLSR, the fusion stacking head) don't have trained weights yet — per
  CLAUDE.md that fine-tuning happens on a rented GPU, not locally — so these currently raise
  `NotImplementedError`. The workers catch that specifically and record a `blocked_on_model_integration`
  status rather than fabricating a score. No plumbing changes will be needed once real weights land.
- `backend/workers/main.py`: entrypoint running all three consumers concurrently
  (`python -m workers.main`).
- `backend/tests/test_scan.py` (+ `conftest.py`): 4 tests against an in-memory sqlite DB with mocked
  storage/queue — upload creates a scan and enqueues the right streams, video vs. image enqueue
  differently, `GET` returns pending status, missing scan 404s.
- **Status:** real infra, verified by tests (`pytest -q` → passing). Per-modality model outputs are
  still placeholders until each detector is actually integrated (see checklist above and in `PLAN.md`).

---

## How to read "Status" going forward

- **real, verified** — actual model/logic runs, backed by a test that exercises it (not mocked away).
- **wired, blocked** — the surrounding plumbing (queue/DB/API) is real; the model call inside it
  legitimately raises `NotImplementedError` because weights aren't trained/integrated yet, and that's
  surfaced as an explicit status rather than a fake number.
- **stub** — not yet started.
