#!/usr/bin/env bash
# GPU-host chain, after h100_graph_engine.sh: (1) agreement of the graph-captured resident path and the compiled cold
# pass with the eager engine on fine-tuned weights (the stock encoder's untrained head gives near-uniform scores, where
# bf16 differences flip decisions), contracts base served masked and mask-trained; (2) cold throughput with each batch's
# host work prepared on a worker thread while the previous batch runs, eager and compiled; (3) the kernel and host
# profile of those runs (where the remaining host time goes). REHEARSAL: DEVICE=cpu DTYPE=fp32 LIMIT=2 TAG=rh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
DEVICE=${DEVICE:-cuda}; DTYPE=${DTYPE:-bf16}; LIMIT=${LIMIT:-0}; TAG=${TAG:-pipe}
log() { echo "[graph2] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
$PY -m pytest -q tests/test_graphs.py tests/test_engine.py 2>&1 | tail -2
log "agreement on fine-tuned weights"
for m in models/_from_h100_lr/modernbert-base-full-contractnli-lr2e5-bestacc:served models/_from_h100_round16/modernbert-base-block-contractnli-steps-s1:masktrained; do
  [ -d ${m%%:*} ] && $PY scripts/graph_engine_bench.py --task contractnli --cell 4096:17 --size base --model ${m%%:*} --device $DEVICE --dtype $DTYPE --limit $LIMIT --tag $TAG-${m#*:} 2>&1 | grep --line-buffered -E "^\[graph\]|Error|Traceback"
done
NS="4 16 60"; [ "$DEVICE" = cpu ] && NS="2"
log "cold throughput with the host work pipelined"
for size in base large; do
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N $NS --size $size --stock --device $DEVICE --dtype $DTYPE --engine-kernel cascade --no-baseline --pipeline --tag $TAG-$size-eager 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N $NS --size $size --stock --device $DEVICE --dtype $DTYPE --engine-kernel cascade --no-baseline --pipeline --compiled --tag $TAG-$size-compiled 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
done
if [ "$DEVICE" = cuda ]; then
  log "profiles: compiled, compiled + pipelined (64 contracts)"
  $PY scripts/engine_profile.py --size base --task contractnli --cell 4096:17 --N 64 --kernel cascade --batch-tokens 65536 --compiled --tag $TAG-compiled 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
  $PY scripts/engine_profile.py --size base --task contractnli --cell 4096:17 --N 64 --kernel cascade --batch-tokens 65536 --compiled --pipeline --tag $TAG-compiled-pipelined 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
fi
log "chain done"
