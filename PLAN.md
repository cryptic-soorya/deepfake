# Morpheus.AI — Build Plan (Clean-Slate Version)

Real-Time AI-Powered Deepfake, Synthetic Media & Digital Identity Verification Platform.
This plan assumes a fresh build using the best available open models as of August 2026 — no legacy code assumptions, no compromises for "what we already have."

Competition: Neurobots Championship 2026 — Robotics, Drones & AI. Four rounds: Idea Validation → Prototype → Product Readiness → Grand Finale.

---

## 1. One-liner

A platform that scores live webcam feeds, video calls, uploaded media, and identity-verification sessions for AI manipulation in real time — face, voice, and lip-sync checked together, every verdict explained in plain language, every decision backed by a signed forensic report.

---

## 2. Model Stack — Best Available, Locked In

No placeholder models. This is the actual stack to build against, chosen for accuracy, generalization to unseen generators, and realistic build time for a small team.

| Capability | Model | Why this one |
|---|---|---|
| Face detection + landmarks | **SCRFD** (InsightFace) | Faster and more accurate than MTCNN/RetinaFace at small/angled faces; outputs 5-point landmarks for free. |
| Fine-grained landmarks / micro-expressions | **MediaPipe Face Mesh** (478 points) | Free, real-time on CPU, standard for micro-expression and head-pose tracking. |
| Face recognition / identity match | **ArcFace** (InsightFace `buffalo_l`) | The production standard face-embedding model; almost every serious face-recognition system in 2026 is trained with ArcFace loss or a variant. |
| Liveness / anti-spoofing | **Silent-Face-Anti-Spoofing** (MiniFASNet) | Open-sourced with full training pipeline, purpose-built for passive liveness, small enough to run on CPU in real time. |
| Frame-level deepfake classifier | **EfficientNet-B4**, trained with the **Self-Blended Images (SBI)** recipe | EfficientNet-B4 was the backbone behind the winning Facebook DFDC Kaggle solution. SBI trains only on blended real images (no fake generator required) and generalizes dramatically better to manipulation methods never seen in training — critical when judges test with their own clips. |
| Video-level / temporal classifier | Lightweight temporal transformer or GRU head over per-frame embeddings | Frame-by-frame averaging misses motion artifacts (unnatural blinking, frame-blend jitter) — one of the strongest deepfake tells. This is the single upgrade that matters most for accuracy. |
| Generalization detector (unseen GANs/diffusion) | **UnivFD** — CLIP ViT-L/14 frozen features + a linear probe | Answers the "detection of unseen generative models" bonus requirement directly. CLIP was never trained on real/fake classification, but its general visual features transfer surprisingly well to both GAN and diffusion-generated content. Almost nobody else at the hackathon will have this. |
| Audio deepfake / voice-clone detector | **AASIST** with a **wav2vec2-XLSR-53** self-supervised front end | AASIST is the strongest reproducible open backbone in the ASVspoof "open condition" leaderboard; XLSR front end adds multilingual, large-scale pretraining. |
| Lip-sync consistency | **SyncNet** (Chung & Zisserman), reused as a detector — optionally cross-checked with ASR vs. VSR (lip-reading) text comparison | Pretrained checkpoint available (reused inside the open-source Wav2Lip repo) — this is inference, not a training project. Scores audio-visual timing distance in a sliding window across the clip. |
| Explainability (visual) | **Grad-CAM / Grad-CAM++** on the frame classifier's last conv layer | Standard, judge-legible XAI technique; cheap to compute. |
| Explanation narrative | **Claude API** (Sonnet) | Turns raw per-modality metrics into a 2–3 paragraph plain-English explanation — a genuine differentiator most hackathon teams skip. |
| Fusion / final score | Small MLP or logistic-regression stacking head over per-modality scores + **temperature scaling** calibration | Learned fusion beats a fixed weighted average, and calibration makes the confidence percentage actually mean something. |

---

## 3. Datasets to Fine-Tune / Benchmark Against

| Modality | Primary training set | Cross-dataset generalization test |
|---|---|---|
| Visual / face | FaceForensics++ + DFDC | Celeb-DF v2, WildDeepfake, FVBench (2025, 42 generative models — best test for "unseen generator" claims) |
| Audio | ASVspoof 2021 (LA + DF) | In-the-Wild (real-world cloned celebrity/politician speech), WaveFake |
| Audio-visual (lip-sync + fusion) | FakeAVCeleb (matched real/fake audio+video pairs) | DeepSpeak v2 |
| Liveness | CelebA-Spoof | CASIA-FASD, OULU-NPU, Replay-Attack, plus your own team's webcam replay attempts (mandatory — public datasets never match your actual camera/lighting) |

---

## 4. Architecture (Target State)

```
[Webcam / Upload / RTSP]
        │
        ▼
[Ingestion Gateway] — WebRTC for live, HTTPS for upload
        │
        ▼
[Preprocessing] — frame sampler, SCRFD face detect+track, audio extraction (16kHz)
        │
        ▼
[Message Queue] — Redis Streams (simplest to stand up fast) or Kafka
        │
        ▼
[Inference Workers — each its own scalable service]
   ├─ Visual classifier (EfficientNet-B4 + SBI, + temporal head)
   ├─ Generalization detector (UnivFD/CLIP)
   ├─ Audio classifier (AASIST + XLSR)
   ├─ Lip-sync scorer (SyncNet)
   └─ Face-ID + liveness (ArcFace + MiniFASNet) — identity flow only
        │
        ▼
[Fusion + Calibration Service] — stacking head, temperature scaling
        │
        ▼
[XAI + Forensic Report Generator] — Grad-CAM, Claude narrative, signed PDF/JSON
        │
        ▼
[API Gateway] — auth, rate limit, audit log
        │
        ▼
[Dashboard] · [Identity Verification Portal] · [External API Consumers]
```

Backend: **FastAPI** (async-native, needed for WebSocket streaming — pick this over Flask from day one). Frontend: React + Vite (already the right choice, keep it). Storage: Postgres (metadata/audit) + S3-compatible object storage (media/reports). Deployment target: Docker containers, GPU-backed inference workers.

---

## 5. Phased Execution Plan (mapped to competition rounds)

### Round 1 — Idea Validation (submit now)
- Problem understanding, tech stack, team roles, GitHub repo — submission content, not build work.
- Lock in the model stack above as the committed technical direction.

### Round 2 — Prototype
Goal: one coherent demo path, even if narrow, using real models from the stack above (not placeholders).
- [x] Stand up FastAPI backend with `/api/v1/scan` (batch upload) end to end — upload → MinIO → Postgres `Scan` row → audit log → Redis Streams → worker consumer group → fusion aggregation is wired and tested (`backend/tests/test_scan.py`). Per-model outputs are still placeholders (see below) until the actual detectors are integrated, so worker output is honestly marked `blocked_on_model_integration` rather than faked. See `PROGRESS.md`.
- [x] Integrate SCRFD face detection — `app/models/face_detector.py` loads real InsightFace weights (`buffalo_l` pack, `detection` module only) and runs genuine ONNX inference; verified in `backend/tests/test_face_detector.py`. See `PROGRESS.md`.
- [ ] Integrate EfficientNet-B4 (SBI-trained) frame classifier + Grad-CAM heatmap.
- [ ] Integrate AASIST + XLSR audio classifier on extracted audio track.
- [ ] Integrate SyncNet lip-sync scorer as a third modality — this is your differentiator, do not skip it.
- [ ] Build the fusion head (even a simple logistic regression trained on a small labeled set beats a fixed weighted average).
- [ ] Wire the Claude explanation endpoint to narrate all three modalities.
- [ ] Frontend: upload flow + score display + heatmap + explanation (extend existing dashboard pattern).
- [ ] Record a benchmark run: AUROC/EER on a held-out slice of FF++/DFDC + ASVspoof, published as numbers, not adjectives.

### Round 3 — Product Readiness
Goal: prove this scales and isn't a toy.
- [ ] Add live webcam path: WebRTC ingestion → message queue → streaming score updates over WebSocket.
- [ ] Add identity verification flow: ArcFace enrollment + match, MiniFASNet passive liveness, one active challenge-response step.
- [ ] Add auth (API keys), rate limiting, and audit logging on every endpoint.
- [ ] Add the UnivFD generalization detector as a secondary "unseen generator" score, surfaced explicitly in the UI — call this out by name to judges.
- [ ] Forensic report generation: signed JSON/PDF per scan.
- [ ] Load-test the streaming path; publish p95 latency numbers.

### Round 4 — Grand Finale
Goal: polish, narrative, and defensibility under judge questioning.
- [ ] Live demo script: one clean happy path (authentic clip), one clean deepfake catch (visual+audio+lip-sync all flagged), one adversarial case (only one modality faked — show the nuanced verdict).
- [ ] One-slide architecture diagram + one-slide benchmark numbers.
- [ ] Be ready to explain, in plain terms, why each model was chosen over the obvious alternative (e.g. why EfficientNet-B4+SBI over a plain fine-tuned CNN, why UnivFD matters for unseen generators) — judges at this level probe exactly this.
- [ ] Stretch/bonus, only if time allows: C2PA watermark verification check, or a documented (even if unimplemented) federated-detection architecture — these are explicitly called out as bonus innovation points.

---

## 6. Benchmark Targets (what "good" looks like)

| Metric | Target |
|---|---|
| Visual detector AUROC (in-distribution, FF++/DFDC held-out) | ≥ 95% |
| Visual detector AUROC (cross-dataset, unseen generator) | ≥ 85% |
| Audio spoof EER (ASVspoof held-out) | ≤ 5% |
| Lip-sync desync classification accuracy | ≥ 90% |
| End-to-end live-scoring latency (p95) | ≤ 300ms per scored window |
| False positive rate on authentic media | ≤ 3% |

---

## 7. Risks

- **Training time on limited hardware.** Fine-tuning EfficientNet-B4/XLSR from scratch on a laptop is slow. Use a free-tier cloud GPU (Colab/Kaggle/Modal) for the actual fine-tuning runs; keep local (M4, MPS) for inference/dev only.
- **Scope creep.** The full stack above is ambitious — if Round 2 is at risk, cut to: visual classifier + audio classifier + Grad-CAM + Claude explanation, and add lip-sync/identity/streaming in Round 3. Never cut the explainability layer — it's cheap and it's a strong differentiator.
- **Model licensing.** Verify license terms on each pretrained checkpoint (InsightFace, AASIST, SyncNet) before claiming "production ready" in judging materials.
