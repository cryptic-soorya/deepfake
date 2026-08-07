# CLAUDE.md — Morpheus.AI

Context file for any AI coding agent (Claude Code or otherwise) working on this project. Read this before writing code. This describes the target build — treat it as the source of truth over any older code lying around elsewhere; this is a clean-slate rebuild.

## What this project is

A real-time platform that detects AI-generated deepfake video, cloned voice, and manipulated identity, and separately verifies a person's digital identity (face match + liveness). Every verdict must come with an explanation, not just a score. Built for the Neurobots Championship 2026 hackathon, but the target architecture is written as if it needs to survive contact with actual enterprise/government use — no shortcuts that would embarrass the team if a judge asks "how does this actually scale."

See `PLAN.md` in this folder for the phased build order and current round targets.

## Locked technical decisions — do not silently change these

- **Backend:** FastAPI (async), not Flask. Streaming/WebSocket support is a first-class requirement, and FastAPI is built for it.
- **Frontend:** React + Vite. Tailwind for styling.
- **Queue:** Redis Streams for the inference pipeline (Kafka only if Redis Streams proves insufficient under load-test — don't reach for Kafka by default, it's more ops overhead than a small team needs).
- **Database:** Postgres for structured data (users, audit logs, scan metadata). Object storage (S3-compatible, e.g. MinIO locally / S3 or R2 in prod) for media files and generated reports — never store binary media in Postgres.
- **Model serving:** start with plain PyTorch/ONNX Runtime inference inside the FastAPI workers; only move to a dedicated model server (Triton/TorchServe) once concurrent load actually requires batching across sessions.

## Model stack (final — see PLAN.md for the reasoning behind each choice)

| Task | Model |
|---|---|
| Face detection + landmarks | SCRFD (InsightFace) |
| Micro-expression / mesh landmarks | MediaPipe Face Mesh |
| Face recognition / identity embedding | ArcFace (InsightFace `buffalo_l`) |
| Liveness / anti-spoofing | Silent-Face-Anti-Spoofing (MiniFASNet) |
| Frame-level deepfake classifier | EfficientNet-B4, trained via the Self-Blended Images (SBI) recipe |
| Video-level temporal classifier | GRU/temporal-attention head over per-frame embeddings |
| Unseen-generator / generalization detector | UnivFD (frozen CLIP ViT-L/14 + linear probe) |
| Audio deepfake / voice-clone detector | AASIST + wav2vec2-XLSR-53 front end |
| Lip-sync consistency | SyncNet (pretrained inference only, not trained from scratch) |
| Visual explainability | Grad-CAM / Grad-CAM++ |
| Narrative explanation | Claude API (Sonnet) |
| Fusion | Small MLP/logistic-regression stacking head + temperature scaling |

Do not substitute a lighter/older model (e.g. MesoNet, plain MTCNN) to save setup time without flagging it — every substitution should be a visible, deliberate tradeoff, not a silent downgrade, because this stack was chosen specifically to be defensible against a technically literate judge.

## Repository structure to build toward

```
/backend
  /app
    main.py                 # FastAPI app entrypoint
    /routers                # scan.py, identity.py, stream.py, explain.py, report.py
    /models                 # model loading/inference wrappers, one file per model in the stack
    /pipeline                # preprocessing: face_track.py, audio_extract.py, frame_sampler.py
    /fusion                  # scoring + calibration
    /reports                  # forensic report generation (PDF/JSON + signing)
  /workers                    # queue consumers, one per inference model
  requirements.txt / pyproject.toml
/frontend
  /src
    /pages                   # Scanner, LiveSession, IdentityPortal, ReportViewer
    /components
/infra
  docker-compose.yml          # local dev: backend, redis, postgres, minio
  /k8s                        # production manifests (later phase)
/docs
  PLAN.md
  CLAUDE.md                   # this file
  benchmark-report.md         # filled in once models are trained/evaluated
```

## Dev environment notes

- Primary dev machine: Apple M4 MacBook Pro, 16GB unified memory. Use **MPS** (`torch.backends.mps`) for local inference/dev, not CUDA — code should detect device via a shared `get_device()` helper (`mps` → `cuda` → `cpu` fallback order) rather than hardcoding.
- 16GB unified memory is fine for running the full inference stack locally at moderate batch size, but **not** for training/fine-tuning EfficientNet-B4 or XLSR from scratch at reasonable speed. Do actual fine-tuning runs on a rented/free-tier GPU (Colab, Kaggle, Modal, RunPod) and pull the resulting checkpoint back down for local inference and demo.
- Keep every model export ONNX-compatible where feasible — this keeps the door open for edge/CPU deployment and quantization later without a rewrite.

## Conventions

- API routes versioned under `/api/v1/...` from the first commit — don't ship unversioned routes and migrate later.
- Every model wrapper exposes a consistent interface: `load()`, `predict(input) -> {score, confidence, raw, metadata}` — this is what makes the fusion layer possible without special-casing each model.
- Every endpoint that touches media or biometric data writes an audit log entry — this isn't optional, add it in the same PR as the endpoint, not as a follow-up.
- Secrets (Anthropic key, DB credentials, etc.) only via environment variables, never committed. `.env.example` should list every required variable with a placeholder value.
- Commit messages describe what changed and why in one line — no "fix stuff" commits.

## What "done" looks like, per phase

Round 2 (Prototype) is done when: a video or image upload returns a fused score across visual + audio + lip-sync, with a Grad-CAM heatmap and a Claude-generated explanation, and you can point at a benchmark number (not an adjective) for at least the visual and audio detectors.

Round 3 (Product Readiness) is done when: the same pipeline also runs on a live webcam session over WebSocket with sub-300ms updates, identity enrollment/verification works end to end, and every request is authenticated and audit-logged.

Round 4 (Grand Finale) is done when: the demo runs three scripted scenarios (authentic, fully-faked, single-modality-faked) without manual intervention, and the team can explain every model choice in the stack table above under questioning.
