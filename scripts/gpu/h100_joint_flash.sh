#!/usr/bin/env bash
# GPU-host chain: the joint-packing latency with the joint sequence also run through variable-length FlashAttention (no
# mask tensor), beside the masked joint path and the cascade engine, in one quiet session.
set -uo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
[ -f data/cache/requests/contractnli.json ] || { echo "missing request cache"; exit 1; }
.venv/bin/python -u -m pytest -q tests/test_joint_flash.py 2>&1 | tail -2
.venv/bin/python -u scripts/joint_packing.py --latency-only --n 50 2>&1 | grep --line-buffered -E "\[joint\]|Error|Traceback"
echo "[joint-flash] chain done"
