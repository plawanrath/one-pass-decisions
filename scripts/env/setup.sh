#!/usr/bin/env bash
# Build the project environment on this machine. Idempotent.
#   bash scripts/env/setup.sh [--docker] [--models] [--fresh]
# Uses the frozen lock for this platform when one exists; otherwise resolves requirements.txt and writes a new lock.
set -euo pipefail
ENV_DIR="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$ENV_DIR/../.." && pwd)"; cd "$REPO"
PYV="$(cat .python-version 2>/dev/null || echo 3.11)"
TAG="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | tr '[:upper:]' '[:lower:]')"
LOCK="$ENV_DIR/requirements.lock.$TAG.txt"
DOCKER=0; MODELS=0; FRESH=0
for a in "$@"; do case "$a" in --docker) DOCKER=1;; --models) MODELS=1;; --fresh) FRESH=1;; esac; done

record() {  # record <key> <seconds>: measured timings feed the handoff time estimate (private, gitignored)
  mkdir -p .research-kit && python3 - "$1" "$2" <<'PY'
import json, pathlib, platform, sys
p = pathlib.Path(".research-kit/timings.json"); d = json.loads(p.read_text()) if p.exists() else {}
d[sys.argv[1]] = int(sys.argv[2]); d["host"] = platform.node(); p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

t0=$(date +%s)
[ "$FRESH" = 1 ] && rm -rf .venv
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then uv venv --python "$PYV" --seed .venv
  else "$(command -v "python$PYV" || command -v python3)" -m venv .venv; fi
fi
pipi() { if command -v uv >/dev/null 2>&1; then uv pip install --python .venv/bin/python "$@"; else .venv/bin/python -m pip install "$@"; fi; }
if [ -f "$LOCK" ]; then
  echo "[setup] installing frozen lock: ${LOCK#$REPO/}"; pipi -q -r "$LOCK"
else
  echo "[setup] no lock for $TAG: resolving requirements.txt, then freezing a new lock"
  pipi -q -r "$ENV_DIR/requirements.txt"; bash "$ENV_DIR/lock.sh"
fi
pipi -q -e . --no-deps
record venv_s $(( $(date +%s) - t0 ))
.venv/bin/python --version

if [ "$DOCKER" = 1 ] && [ -f "$ENV_DIR/docker-compose.yml" ]; then
  t0=$(date +%s); docker compose -f "$ENV_DIR/docker-compose.yml" build
  docker compose -f "$ENV_DIR/docker-compose.yml" up -d; record docker_build_s $(( $(date +%s) - t0 ))
fi
if [ "$MODELS" = 1 ] && [ -f "$ENV_DIR/pin_models.py" ]; then
  t0=$(date +%s); .venv/bin/python "$ENV_DIR/pin_models.py" --from-lock; record models_s $(( $(date +%s) - t0 ))
fi
t0=$(date +%s); .venv/bin/python -m pytest -q -m "not slow and not gpu and not docker" || echo "[setup] tests FAILED"
record verify_s $(( $(date +%s) - t0 ))
