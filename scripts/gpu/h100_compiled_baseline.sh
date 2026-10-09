#!/usr/bin/env bash
# GPU-host chain (timing pod, nothing else on the GPU): the unpadded baseline with the same compiled attention-free
# pieces as the engine, against the eager baseline and the eager and compiled engine, in one session with every
# request shape served once before timing on every path (tag strength-compiled): contracts 4,096 and 2,048 tokens with
# 17 hypotheses (base) and 4,096 (large). REHEARSAL: positional "rehearse" (CPU, fp32, 2 requests per cell).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
DEV="--device cuda --dtype bf16 --no-cudnn-sdp"; N=50; TAG=strength-compiled; SIZES="base large"
if [ "${1:-}" = rehearse ]; then DEV="--device cpu --dtype fp32"; N=1; TAG=strength-compiled-rh; SIZES=base; fi
log() { echo "[compiled-baseline] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/test.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
$PY -m pytest -q tests/test_round19.py tests/test_graphs.py tests/test_varlen.py 2>&1 | tail -2
lat() { $PY scripts/latency_sweep.py $DEV --stock --warm-shapes --task contractnli --n $N --warmup 3 "$@" 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"; }
for size in $SIZES; do
  cells="4096:17"; [ $size = base ] && cells="2048:17 4096:17"
  [ "${1:-}" = rehearse ] && cells="2048:17"
  log "$size: baseline eager and compiled"
  lat --size $size --cells $cells --paths varlen-batched varlen-batched-compiled --out-tag $TAG
  log "$size: engine eager"
  lat --size $size --cells $cells --paths engine-cold-fill engine-hit --engine-kernel cascade --out-tag $TAG
  log "$size: engine compiled"
  lat --size $size --cells $cells --paths engine-cold-fill engine-hit --engine-kernel cascade --engine-compiled --out-tag $TAG
done
log "chain done"
