#!/usr/bin/env bash
# Starts the entire Morpheus.AI stack for local development:
#   - infra containers (postgres, redis, minio) via docker compose
#   - backend API (uvicorn)
#   - background inference workers (python -m workers.main)
#   - frontend dev server (vite)
#
# Usage:
#   ./run.sh          start everything, stream logs, Ctrl+C to stop
#   ./run.sh stop     stop everything this script started (infra keeps running)
#   ./run.sh down     stop everything, including the infra containers
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
INFRA_DIR="$ROOT_DIR/infra"
LOG_DIR="$ROOT_DIR/.run"
PID_FILE="$LOG_DIR/pids"

mkdir -p "$LOG_DIR"

log() { printf "\033[1;36m[run]\033[0m %s\n" "$1"; }
err() { printf "\033[1;31m[run]\033[0m %s\n" "$1" >&2; }

stop_app_processes() {
    if [[ -f "$PID_FILE" ]]; then
        log "stopping backend / workers / frontend..."
        while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
}

if [[ "${1:-}" == "stop" ]]; then
    stop_app_processes
    log "infra containers left running -- use './run.sh down' to stop those too."
    exit 0
fi

if [[ "${1:-}" == "down" ]]; then
    stop_app_processes
    log "stopping infra containers..."
    (cd "$INFRA_DIR" && docker compose down)
    exit 0
fi

# --- 0. sanity checks -------------------------------------------------------

if [[ ! -f "$ROOT_DIR/.env" ]]; then
    err ".env not found at $ROOT_DIR/.env -- copy .env.example and fill in secrets first."
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    err "Docker daemon isn't running -- start Docker Desktop first."
    exit 1
fi

# --- 1. infra: postgres, redis, minio ---------------------------------------

log "starting infra containers (postgres, redis, minio)..."
(cd "$INFRA_DIR" && docker compose up -d postgres redis minio)

log "waiting for postgres..."
until docker exec infra-postgres-1 pg_isready -U morpheus > /dev/null 2>&1; do sleep 1; done
log "waiting for redis..."
until docker exec infra-redis-1 redis-cli ping > /dev/null 2>&1; do sleep 1; done
log "infra is up."

# --- 2. backend venv / frontend node_modules --------------------------------

if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
    log "creating backend venv + installing requirements (first run only)..."
    python3 -m venv "$BACKEND_DIR/.venv"
    "$BACKEND_DIR/.venv/bin/pip" install --quiet -r "$BACKEND_DIR/requirements.txt"
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    log "installing frontend dependencies (first run only)..."
    (cd "$FRONTEND_DIR" && npm install)
fi

# --- 3. load env, start app processes ---------------------------------------

set -a
source "$ROOT_DIR/.env"
set +a

: > "$PID_FILE"

log "starting background inference workers..."
(cd "$BACKEND_DIR" && exec "$BACKEND_DIR/.venv/bin/python" -m workers.main) > "$LOG_DIR/workers.log" 2>&1 &
echo $! >> "$PID_FILE"

log "starting backend API (uvicorn)..."
(cd "$BACKEND_DIR" && exec "$BACKEND_DIR/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000) > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" >> "$PID_FILE"

log "starting frontend (vite)..."
(cd "$FRONTEND_DIR" && exec npm run dev -- --host) > "$LOG_DIR/frontend.log" 2>&1 &
echo $! >> "$PID_FILE"

trap 'stop_app_processes; exit 0' INT TERM

log "waiting for backend model warmup (see .run/backend.log for detail)..."
for _ in $(seq 1 120); do
    if curl -s http://localhost:8000/api/v1/health 2>/dev/null | grep -q '"models_ready":true'; then
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        err "backend process died -- check $LOG_DIR/backend.log"
        stop_app_processes
        exit 1
    fi
    sleep 1
done

log "everything is up:"
log "  frontend:  http://localhost:5173"
log "  backend:   http://localhost:8000  (docs at /docs)"
log "  minio:     http://localhost:9001  (minioadmin / minioadmin)"
log "logs: $LOG_DIR/{backend,workers,frontend}.log"
log "Ctrl+C to stop backend/workers/frontend (infra containers keep running)."

wait
