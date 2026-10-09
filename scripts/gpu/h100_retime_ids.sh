#!/usr/bin/env bash
# GPU-host chain: the main-text latency cells re-timed with every contract tokenized once outside the timed region on
# every path (an earlier sweep tokenized inside the baselines' timed calls while the engine reused cached token ids).
# Baseline and cascade engine in one session: the K sweep (base, large), the 4k and 2k contract cells, one clause window,
# four windows (split engine), and the QuALITY-shaped cell. Output tag "idsfix"; earlier files stay as they are.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[idsfix] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
TAG=idsfix
for t in contractnli cuad quality; do [ -f data/pools/$t/test.jsonl ] || { log "missing data/pools/$t"; exit 1; }; done
lat() { $PY scripts/latency_sweep.py --device cuda --dtype bf16 --no-cudnn-sdp --stock "$@" 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"; }
log "contracts: K sweep and 2k/4k cells, base and large"
for size in base large; do
  lat --size $size --task contractnli --n 50 --cells 2048:17 4096:17 --paths varlen-batched --out-tag strength-$TAG
  lat --size $size --task contractnli --n 50 --cells 2048:17 4096:17 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
  lat --size $size --task contractnli --n 30 --cells 4096:1 4096:2 4096:5 4096:10 --paths varlen-batched --out-tag strength-$TAG
  lat --size $size --task contractnli --n 30 --cells 4096:1 4096:2 4096:5 4096:10 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
done
log "contracts 4k/17: the other full-attention baselines (appendix)"
lat --size base --task contractnli --n 20 --cells 4096:17 --paths one-seq compile-one-seq varlen-one-seq naive-batch --out-tag strength-$TAG
log "clauses and stories"
lat --size base --task cuad --n 30 --cells 8192:41 --paths varlen-batched --out-tag strength-$TAG
lat --size base --task cuad --n 30 --cells 8192:41 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
lat --size base --task cuad --n 20 --cells 32768:41 --paths varlen-batched engine-windowed-cold engine-windowed-hit --engine-kernel split --out-tag strength-$TAG-windowed
lat --size base --task quality --n 30 --cells 8192:17 --paths varlen-batched --out-tag strength-$TAG
lat --size base --task quality --n 30 --cells 8192:17 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
log "chain done"
