# PROGRESS.md — Morpheus.AI Build Log

Running changelog of what's been built, by whom, and what's still stubbed. See `PLAN.md` for the phased
target and `CLAUDE.md` for the locked architecture decisions this work follows. Newest entries first.

Convention: entries note **what changed**, **why**, and honestly flag anything still faked/stubbed —
per CLAUDE.md, no silent placeholders pretending to be real results.

---

## 2026-08-08 — Grad-CAM heatmap + narrative explainer (Gemini, not Claude)

**Who:** cryptic-soorya (via Claude Code)
**Round:** 2 (Prototype)

- `backend/app/models/gradcam.py`: `GradCAMExplainer` was `NotImplementedError`; now
  reuses `EfficientNetB4SBI`'s loaded checkpoint and `_conv_head` (the last conv layer
  before pooling) as the Grad-CAM target. Runs a real forward pass + backward pass on the
  P(fake) logit, weights the target layer's activations by pooled gradients, ReLUs and
  normalizes into a CAM, resizes to the 380x380 crop, and overlays it (JET colormap) on
  the face crop as a base64 PNG. Deliberately shares the classifier's own face crop
  instead of loading a second copy of the network, so the heatmap and the score it
  explains are computed from the same input.
- `backend/workers/frame_classifier_worker.py`: `_aggregate_frame_predictions` now also
  returns the winning frame's raw image (`_best_frame_image`, popped off before the
  aggregate dict is persisted — ScanResult.raw is a JSON column, can't hold an ndarray).
  `FrameClassifierWorker.handle` runs Grad-CAM on that frame (or the raw bytes, for
  single-image scans) after the classifier call, uploads the heatmap PNG to object
  storage as `gradcam/{scan_id}.png` (per CLAUDE.md: no binary media in Postgres), and
  persists a second `ScanResult` row (`model_name="gradcam"`) holding the score/CAM
  array/heatmap *key*, not the PNG bytes. Missing checkpoint or no-face-detected cases
  degrade the same way the classifier itself does (`blocked`/`no face detected`), not a
  crash.
- **Narrative explainer — deliberate model substitution, flagged per CLAUDE.md:** team
  explicitly asked to use **Gemini 3.1 Flash-Lite** instead of the Claude API (Sonnet)
  CLAUDE.md's model-stack table specifies for this slot. This is not a silent downgrade —
  CLAUDE.md's table has been updated to record the swap and the reason (cost/latency for
  a narrative task that doesn't need Sonnet-level reasoning), and it's called out here.
  `backend/app/models/claude_explainer.py` deleted; replaced by
  `backend/app/models/gemini_explainer.py` (`GeminiExplainer`), using the `google-genai`
  SDK. Reads `GEMINI_API_KEY` (required) and `GEMINI_MODEL` (optional, defaults to
  `gemini-3.1-flash-lite`) from the environment. `_build_prompt` turns the fused verdict +
  per-modality scores/metadata into a plain-English prompt; `predict()` returns the
  narrative in `metadata.narrative`.
- `backend/app/routers/explain.py`: now imports `GeminiExplainer`; the except clause
  broadened from just `NotImplementedError` to also catch `KeyError` (API key not set)
  and `google.genai.errors.APIError` (key set but rejected — e.g. the `.env.example`
  placeholder value — or a Gemini-side failure), all mapping to the existing
  `blocked_on_model_integration` / `501` path rather than a 500 crash.
- `backend/app/config.py`, `.env.example`, `backend/tests/conftest.py`: `ANTHROPIC_API_KEY`
  → `GEMINI_API_KEY` (+ new optional `GEMINI_MODEL`) throughout.
- `backend/requirements.txt`: `anthropic` → `google-genai`.
- `backend/tests/test_gradcam.py`: integration test against the real SBI checkpoint
  (module-scoped fixture, skips offline) — asserts a well-formed score/confidence and a
  decodable 380x380 PNG heatmap on a real photo, and the no-face path on a blank image.
- `backend/tests/test_gemini_explainer.py`: unit tests with a mocked `genai.Client` (no
  live `GEMINI_API_KEY` in this environment) — prompt content, response shaping, and the
  missing-key `KeyError` path.
- Verified locally: `pytest -q` → 55 passed (was 50), including the Grad-CAM test
  actually downloading and running the real EfficientNet-B4/SBI checkpoint (not skipped).
- **Update, same day:** team dropped two live Gemini keys into `.env`. Added key-fallback
  to `GeminiExplainer`: `load()` builds a client per configured key (`GEMINI_API_KEY`
  required, `GEMINI_API_KEY_2` optional), `predict()` tries each in order and returns on
  first success, re-raising the last error (still an `APIError`, so `explain.py`'s
  handling is unchanged) only if every key fails. `metadata.key_used` records which key
  answered (`"primary"`/`"fallback"`) so an operator can tell from a scan's stored
  explanation whether the primary key was degraded at request time. Reason: two keys so a
  single key's rate limit doesn't stall scans mid-demo.
  `backend/tests/test_gemini_explainer.py` covers primary-success, fallback-on-error, and
  both-keys-fail (mocked). **Then verified live**, not just mocked: ran
  `GeminiExplainer.predict()` against the real primary key (real narrative came back,
  good quality, confirms `gemini-3.1-flash-lite` is in fact a real, callable model id) and
  again with the primary key swapped for an invalid one to force a genuine `APIError` and
  confirm the real secondary key answers (`key_used: fallback`). Both ad hoc, not part of
  the committed test suite (no real keys in CI) — pytest suite is still mock-only by
  design. `pytest -q` → 57 passed (was 55).
- **Status:** real, verified for Grad-CAM (genuine forward+backward pass, tested against
  real weights) and now for the Gemini explainer + fallback too — both the primary and
  fallback paths were exercised against live Gemini keys, not just mocked in the test
  suite.
- **Not done yet:** AASIST/XLSR audio classifier and SyncNet lip-sync wrappers exist with
  real code (per earlier sessions) but weren't touched or re-verified here. The fusion
  head is still the interim equal-weighted average — Grad-CAM and the explainer don't feed
  into fusion, they're presentation/explainability layers. Temporal classifier and UnivFD
  are still stub.

## 2026-08-08 — Video frame sampling + interim fusion head

**Who:** cryptic-soorya (via Claude Code)
**Round:** 2 (Prototype)

- `backend/app/pipeline/frame_sampler.py`: `sample_frames()` was `NotImplementedError`; now
  decodes a video via `cv2.VideoCapture` and yields BGR frames downsampled to a target fps
  (falls back to yielding every frame if the container's own fps can't be read).
- `backend/workers/frame_classifier_worker.py`: previously fed raw video bytes straight into
  `EfficientNetB4SBI.predict()`, which expects a single image — video scans could not actually
  reach the classifier. Now branches on `media_type`: video uploads get written to a temp file,
  sampled at 2fps via `sample_frames()` (capped at 30 frames/scan to bound per-scan compute on a
  laptop-class worker), scored frame-by-frame, and aggregated by mean score/confidence. Flagged
  explicitly in the docstring as an interim aggregation — it's mean pooling, not the
  GRU/temporal-attention head PLAN.md calls for, so motion-artifact tells (blink irregularities,
  frame-blend jitter) aren't caught yet. Image uploads are unchanged (single `predict()` call).
- `backend/app/fusion/scoring.py`: `fuse()` was `NotImplementedError`; now an equal-weighted
  average over each available modality's score, direction-normalized onto a common "P(fake)"
  axis first (`audio_deepfake`'s score is P(real/bonafide), so it's inverted before averaging).
  Modalities with a `None` score (blocked/no-detection) are skipped rather than counted as
  evidence; raises `NotImplementedError` if literally no modality produced a score, so
  `fusion_worker.py`'s existing `except NotImplementedError` path still records `blocked_on_model_integration`
  instead of a fake verdict. **Honesty note:** this is explicitly *not* the trained
  logistic-regression/MLP stacking head + temperature scaling PLAN.md specifies — no labeled
  fusion training set exists yet. Docstring and `metadata.method` both say
  `equal_weighted_average_interim_pending_stacking_head` so this can't be mistaken for the real
  thing later.
- `backend/tests/test_frame_sampler.py`: synthetic 10fps/20-frame video via `cv2.VideoWriter`,
  asserts downsampling math (10fps→2fps = every 5th frame) and the missing-file error path.
- `backend/tests/test_fusion_scoring.py`: agreement (both fake / both authentic), direction
  inversion for `audio_deepfake`, skipping a blocked modality, and the all-`None` → `NotImplementedError`
  path.
- Verified locally: `pytest -q` → 35 passed (was 28).
- **Status:** real, verified — video scans can now actually reach the frame classifier; scan
  pipeline can now produce a genuine fused verdict end-to-end for image/video uploads where the
  underlying detectors report a score. Still explicitly interim per above; upgrade path is a
  trained stacking head once labeled fusion data exists, and a temporal head for video aggregation.
- **Not done yet:** `pipeline/face_track.py` and `pipeline/audio_extract.py` are still
  `NotImplementedError` (audio currently goes through `AASISTVoiceDetector`'s own librosa-based
  loader instead, unaffected by this change). Grad-CAM, the Claude explainer, UnivFD, and the
  temporal classifier are all still stub.

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
