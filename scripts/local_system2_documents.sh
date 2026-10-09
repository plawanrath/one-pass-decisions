#!/usr/bin/env bash
# The System 2 model on the document tasks (MLX; nothing else on the GPU meanwhile), then the risk-controlled thresholds and the
# cascade on long documents.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
log() { echo "[s2docs] $(date '+%F %T') $*"; }
log "System 2 on contracts"
$PY scripts/system2_documents.py --task contractnli --splits test cal 2>&1 | grep --line-buffered -E "accuracy|/|Error|Traceback"
log "System 2 on clauses"
$PY scripts/system2_documents.py --task cuad --splits test cal 2>&1 | grep --line-buffered -E "accuracy|/|Error|Traceback"
$PY scripts/risk_long.py 2>&1 | grep -v skip
$PY scripts/cascade_long.py 2>&1 | grep -v skip
log "done"
