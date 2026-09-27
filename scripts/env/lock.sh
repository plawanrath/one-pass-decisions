#!/usr/bin/env bash
# Freeze the current venv into the lock for this platform (no editable line, so no local paths leak into the repo).
set -euo pipefail
ENV_DIR="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$ENV_DIR/../.." && pwd)"
TAG="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | tr '[:upper:]' '[:lower:]')"
OUT="$ENV_DIR/requirements.lock.$TAG.txt"
{ echo "# Frozen from a real install on $TAG, $("$REPO/.venv/bin/python" --version). Regenerate: bash scripts/env/lock.sh"
  "$REPO/.venv/bin/python" -m pip freeze --exclude-editable; } > "$OUT"
echo "[lock] wrote ${OUT#$REPO/} ($(grep -vc '^#' "$OUT") packages)"
