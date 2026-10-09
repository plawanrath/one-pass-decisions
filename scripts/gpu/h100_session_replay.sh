#!/usr/bin/env bash
# GPU-host chain: reuse across requests on two documented review patterns (scripts/session_replay.py), every request
# served once on every path before timing, with the GPU to itself.
set -uo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
for t in contractnli cuad; do [ -f data/pools/$t/test.jsonl ] || { echo "missing data/pools/$t"; exit 1; }; done
.venv/bin/python -u scripts/session_replay.py --device cuda --contracts 30 2>&1 | grep --line-buffered -E "\[session\]|Error|Traceback"
echo "[session-replay] chain done"
