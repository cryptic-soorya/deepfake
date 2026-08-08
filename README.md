# Morpheus.AI

Real-time deepfake video/voice/identity detection with explainable verdicts, built for the Neurobots Championship 2026.



## Repo layout

```
/backend    FastAPI app, model wrappers, pipeline, fusion, reports, queue workers
/frontend   React + Vite + Tailwind
/infra      docker-compose (local dev) + k8s manifests (prod, later)
/docs       plan, architecture notes, benchmark report
```

## Getting started

1. Copy `.env.example` to `.env` and fill in real values (get secrets from whoever set up the Anthropic key / S3 bucket — never commit `.env`).
2. Bring up the stack:
   ```
   cd infra
   docker compose up
   ```
   This starts Postgres, Redis, MinIO, the backend (`localhost:8000`), and the frontend (`localhost:5173`).
3. Backend-only local dev (no Docker):
   ```
   cd backend
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
4. Frontend-only local dev:
   ```
   cd frontend
   npm install
   npm run dev
   ```

## Conventions

- All API routes live under `/api/v1/...`.
- Every model wrapper implements `load()` / `predict(input) -> {score, confidence, raw, metadata}` (see `backend/app/models/base.py`).
- Any endpoint touching media or biometric data must write an audit log entry (`backend/app/audit.py`) in the same PR.
- Secrets only via environment variables — never commit `.env`.

## Team workflow

- Branch off `main`, open a PR, at least one review before merge.
- Keep commits scoped: one logical change per commit, message explains why not just what.
