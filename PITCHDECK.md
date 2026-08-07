# Morpheus.AI — Pitch Deck (Round 1: Idea Validation)

Neurobots Championship 2026 — Robotics, Drones & AI
Draft as of 2026-08-07. Slide-by-slide speaker content below; adapt line lengths to whatever deck template you use.

---

## Slide 1 — Title

**Morpheus.AI**
Real-Time Deepfake, Synthetic Media & Digital Identity Verification

- Team name / member names / roles
- Neurobots Championship 2026 — Round 1: Idea Validation
- Tagline: "Every verdict, explained — not just scored."

---

## Slide 2 — The Problem

- AI-generated video, cloned voice, and manipulated identity are now cheap and fast to produce — real-time face swaps and voice clones are consumer-grade tools in 2026, not research demos.
- Deepfakes are already being used for financial fraud (fake video-call approvals), disinformation, and identity-verification bypass (KYC fraud at banks/fintechs).
- Existing detectors are black boxes: they output a score with no explanation, so a bank compliance officer or a journalist can't act on the result with confidence — and most fail badly on generators they weren't trained on.
- **The gap:** nobody combines multi-modal detection (visual + audio + lip-sync) with plain-language explainability and identity verification in one real-time platform.

---

## Slide 3 — The Solution

**Morpheus.AI** scores live webcam feeds, video calls, and uploaded media for AI manipulation in real time — checking face, voice, and lip-sync together — and explains every verdict in plain English, backed by a signed forensic report.

Three pillars:
1. **Detect** — is this media AI-generated or manipulated?
2. **Explain** — why did the system flag it, in language a non-technical judge/compliance officer/journalist can understand?
3. **Verify** — is this actually the person they claim to be? (face match + liveness)

---

## Slide 4 — How It Works (High-Level Flow)

```
Webcam / Video Call / Upload
        ↓
Face + Audio Extraction
        ↓
Parallel Detection: Visual  |  Audio  |  Lip-Sync  |  Unseen-Generator Check
        ↓
Fusion Engine → single calibrated confidence score
        ↓
Grad-CAM Heatmap + Claude-Generated Plain-English Explanation
        ↓
Signed Forensic Report (JSON/PDF) + Dashboard
```

- Every stage is a swappable, independently scalable service — not a monolith.
- Same pipeline serves both **batch upload** and **live streaming** (WebSocket) use cases.

---

## Slide 5 — What Makes This Different (Differentiators)

- **Multi-modal fusion, not single-signal detection** — most hackathon/commercial tools check video OR audio. We cross-check visual, audio, AND lip-sync consistency together, which catches the single-modality-faked cases others miss.
- **Explainability is a first-class feature, not an afterthought** — Grad-CAM visual heatmaps + a Claude-generated narrative explanation, so a verdict is defensible, not just a percentage.
- **Built for generators the model has never seen** — a dedicated generalization detector (frozen CLIP features + linear probe) specifically targets the "your model only catches what it was trained on" failure mode — this is the #1 way deepfake detectors fail in the real world.
- **Detection + identity verification in one platform** — most tools do one or the other; we do both, so the same system that flags a deepfake video call can also verify the caller is who they claim to be.
- **Forensic-grade output** — signed reports suitable for compliance/audit trails, not just a demo score on a webpage.

---

## Slide 6 — Technical Approach (Model Stack)

| Capability | Model | Why |
|---|---|---|
| Face detection + landmarks | SCRFD (InsightFace) | Fast, accurate on small/angled faces |
| Face recognition / identity | ArcFace (`buffalo_l`) | Production-standard face embedding |
| Liveness / anti-spoofing | Silent-Face-Anti-Spoofing (MiniFASNet) | Real-time passive liveness, runs on CPU |
| Frame-level deepfake classifier | EfficientNet-B4 + Self-Blended Images training | Backbone behind the winning DFDC solution; SBI generalizes to unseen manipulations |
| Video-level temporal classifier | GRU / temporal-attention head | Catches motion artifacts (blink patterns, frame-blend jitter) that frame-by-frame analysis misses |
| Unseen-generator detector | UnivFD (frozen CLIP ViT-L/14 + linear probe) | Directly targets generalization to GANs/diffusion models never seen in training |
| Audio deepfake / voice clone | AASIST + wav2vec2-XLSR-53 | Strongest reproducible open backbone on ASVspoof leaderboard |
| Lip-sync consistency | SyncNet (pretrained) | Audio-visual timing mismatch is a strong, cheap deepfake tell |
| Visual explainability | Grad-CAM / Grad-CAM++ | Judge-legible heatmap of what the model looked at |
| Narrative explanation | Claude API (Sonnet) | Converts raw metrics into a plain-English explanation |
| Fusion | MLP/logistic-regression stacking + temperature scaling | Learned fusion beats fixed weighted averages; calibration makes the confidence number meaningful |

*(Full reasoning per model choice lives in the team's technical plan — happy to go deeper on any row.)*

---

## Slide 7 — System Architecture

```
[Webcam / Upload / RTSP]
        │
        ▼
[Ingestion Gateway] — WebRTC (live) / HTTPS (upload)
        │
        ▼
[Preprocessing] — frame sampling, face tracking, audio extraction
        │
        ▼
[Redis Streams Queue]
        │
        ▼
[Inference Workers] — Visual | Audio | Lip-Sync | Generalization | Face-ID/Liveness
        │
        ▼
[Fusion + Calibration Service]
        │
        ▼
[XAI + Forensic Report Generator]
        │
        ▼
[API Gateway — auth, rate limit, audit log]
        │
        ▼
[Dashboard] · [Identity Verification Portal] · [External API Consumers]
```

- **Backend:** FastAPI (async, WebSocket-native)
- **Frontend:** React + Vite, Tailwind
- **Data:** Postgres (structured/audit) + S3-compatible object storage (media/reports)
- **Queue:** Redis Streams (Kafka only if load-testing proves it necessary)

---

## Slide 8 — Datasets & Benchmarking Plan

| Modality | Training | Cross-generalization test |
|---|---|---|
| Visual/face | FaceForensics++ + DFDC | Celeb-DF v2, WildDeepfake, FVBench (42 generative models) |
| Audio | ASVspoof 2021 (LA + DF) | In-the-Wild, WaveFake |
| Audio-visual / lip-sync | FakeAVCeleb | DeepSpeak v2 |
| Liveness | CelebA-Spoof | CASIA-FASD, OULU-NPU, Replay-Attack + our own webcam replay tests |

**Target benchmarks (not adjectives):**
- Visual AUROC (in-distribution): ≥ 95%
- Visual AUROC (unseen generator): ≥ 85%
- Audio spoof EER: ≤ 5%
- Lip-sync desync accuracy: ≥ 90%
- End-to-end live latency (p95): ≤ 300ms
- False positive rate on authentic media: ≤ 3%

---

## Slide 9 — Roadmap (Competition Rounds)

- **Round 1 (now) — Idea Validation:** problem framing, locked model stack, architecture, team roles.
- **Round 2 — Prototype:** batch upload pipeline live — visual + audio + lip-sync detection, Grad-CAM heatmap, Claude explanation, benchmark numbers published.
- **Round 3 — Product Readiness:** live webcam scoring over WebSocket (sub-300ms), identity enrollment/verification, auth + audit logging on every endpoint, unseen-generator score surfaced in UI.
- **Round 4 — Grand Finale:** three scripted live demos (authentic / fully faked / single-modality faked), full architecture + benchmark walkthrough, ready to defend every model choice under judge questioning.

---

## Slide 10 — Real-World Use Cases

- **Fintech/banking KYC** — stop deepfake video-call fraud during remote account opening or high-value transaction approval.
- **Enterprise video conferencing** — flag manipulated participants in real time during sensitive calls (M&A, legal, board meetings).
- **Media/journalism verification** — forensic report on submitted video/audio evidence before publication.
- **Government/legal** — court-admissible-style signed reports for evidentiary media review.

---

## Slide 11 — Team

- [Name] — [Role, e.g. ML/backend lead]
- [Name] — [Role, e.g. frontend/product]
- [Name] — [Role, e.g. infra/deployment]
- *(Fill in actual names, roles, and one line of relevant background each.)*

---

## Slide 12 — Ask / Closing

- What we're asking judges to evaluate us on: technical defensibility, explainability, and real-world deployability — not just a demo trick.
- Closing line: "Deepfake detection that doesn't just say 'fake' — it shows its work."
- Contact / GitHub repo link.

---

### Notes for whoever builds the slides
- Slides 6–8 are the most "judge will probe this" content — keep the model-choice table and benchmark targets visible/quotable, since PLAN.md commits to defending every substitution.
- Slide 9's phase gates map directly to `PLAN.md` — update this slide as rounds are completed rather than rewriting it.
- No benchmark numbers are real yet (Round 2 not started) — Slide 8 targets are goals, present them as such, not achieved results.
