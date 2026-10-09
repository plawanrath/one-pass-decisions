#!/usr/bin/env bash
# GPU-host chain: the engine with launch overhead removed. The resident path as CUDA graphs per (state, question)
# bucket and the cold pass with its attention-free pieces compiled, against the eager engine in the same process:
# latency and agreement on every request (contracts base and large at 4k/17, clauses at 8k/41), resident throughput,
# cold throughput (N = 1, 4, 16, 64; eager and compiled), and the kernel profile of the compiled cold pass.
# The GPU runs nothing else during this chain. REHEARSAL: DEVICE=cpu DTYPE=fp32 LIMIT=2 TAG=rh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
DEVICE=${DEVICE:-cuda}; DTYPE=${DTYPE:-bf16}; LIMIT=${LIMIT:-0}; TAG=${TAG:-graph}
log() { echo "[graph] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for t in contractnli cuad; do [ -f data/cache/requests/$t.json ] || { log "missing data/cache/requests/$t.json"; exit 1; }; done
log "tests (CUDA graph capture included)"
$PY -m pytest -q tests/test_graphs.py tests/test_engine.py tests/test_fused_trim.py tests/test_varlen.py 2>&1 | tail -3
B="--device $DEVICE --dtype $DTYPE --stock --limit $LIMIT --tag $TAG"
log "latency, agreement, resident throughput"
$PY scripts/graph_engine_bench.py --task contractnli --cell 4096:17 --size base $B 2>&1 | grep --line-buffered -E "^\[graph\]|Error|Traceback"
$PY scripts/graph_engine_bench.py --task contractnli --cell 4096:17 --size large $B 2>&1 | grep --line-buffered -E "^\[graph\]|Error|Traceback"
$PY scripts/graph_engine_bench.py --task cuad --cell 8192:41 --size base $B 2>&1 | grep --line-buffered -E "^\[graph\]|Error|Traceback"
NS="1 4 16 64"; [ "$DEVICE" = cpu ] && NS="2"
log "cold throughput, eager and compiled"
for size in base large; do
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N $NS --size $size --stock --device $DEVICE --dtype $DTYPE --engine-kernel cascade --no-baseline --tag $TAG-$size-eager 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N $NS --size $size --stock --device $DEVICE --dtype $DTYPE --engine-kernel cascade --no-baseline --compiled --tag $TAG-$size-compiled 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
done
if [ "$DEVICE" = cuda ]; then
  log "kernel profile of the compiled cold pass (64 contracts)"
  $PY scripts/engine_profile.py --size base --task contractnli --cell 4096:17 --N 64 --kernel cascade --batch-tokens 65536 --compiled --tag $TAG-compiled 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
  $PY scripts/engine_profile.py --size base --task contractnli --cell 4096:17 --N 64 --kernel cascade --batch-tokens 65536 --tag $TAG-eager 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
fi
log "chain done"
