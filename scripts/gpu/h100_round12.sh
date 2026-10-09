#!/usr/bin/env bash
# GPU-host chain for the round-12 review: (0) baseline throughput re-measured with one tokenization per contract (the
# earlier throughput runs re-tokenized the contract once per question inside the timed region), GPU exclusive;
# (1) the multi-head baseline on contracts and clauses, the held-out-question arm on contracts and the DeFormer-style
# decoupled arm at l = 11; (2) the decoupled arm at l = 7 and l = 15 and the multi-head baseline on ECtHR. Base-size
# jobs only, at most four on the GPU at a time (about 20 GB each). REHEARSAL: EXTRA / MHEXTRA / TAGSFX as in h100_adapt.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; MHEXTRA=${MHEXTRA:-}; TAGSFX=${TAGSFX:-}; SKIPTP=${SKIPTP:-}
log() { echo "[round12] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
lock() {
  local t=$1
  [ -f data/pools/$t/test.jsonl ] && return 0
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/$t/POOL.json'))['manifest_sha256'])")
  mv data/pools/$t/POOL.json data/pools/$t/POOL.tracked.json
  $PY scripts/lock_pools.py --task $t || { log "$t pool lock FAILED"; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/$t/POOL.json'))['manifest_sha256'])" 2>/dev/null)
  if [ "$want" = "$got" ]; then rm data/pools/$t/POOL.tracked.json; log "$t pools verified"; else log "$t manifest mismatch"; mv data/pools/$t/POOL.tracked.json data/pools/$t/POOL.json; fi
}
mh() {  # task args...
  local t=$1; shift
  [ -f results/multihead/base-$t$TAGSFX/summary.json ] || $PY scripts/multihead_finetune.py --task $t --size base "$@" $MHEXTRA ${TAGSFX:+--tag ${TAGSFX#-}} 2>&1 | grep --line-buffered -v "it/s\]" > logs/multihead_$t$TAGSFX.log
  log "done multihead $t: $(.venv/bin/python -c "import json;d=json.load(open('results/multihead/base-$t$TAGSFX/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4), 'ms/doc', d.get('latency_ms_per_document'))" 2>/dev/null)"
}
ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), 'held-out', round(s.get('test_heldout',{}).get('accuracy',float('nan')),4), 'held-out served-masked', round(s.get('test_heldout_as_block',{}).get('accuracy',float('nan')),4))" 2>/dev/null)"
}
DEC="--size base --mode decoupled --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy"
if [ -z "$SKIPTP" ]; then
  log "phase 0: throughput with one tokenization per contract (engine and unpadded baseline, GPU exclusive)"
  for size in base large; do
    $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size $size --stock --device cuda --dtype bf16 --engine-kernel cascade --tag tokfix-$size-cascade 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
  done
fi
lock cuad
lock ecthr
log "phase 1: multi-head (contracts, clauses), held-out questions, decoupled l=11"
mh contractnli --epochs 5 --select-points 10 &
mh cuad --epochs 3 --tokens-per-batch 32768 --max-docs 4 --select-points 5 &
ft base-full-contractnli-holdout$TAGSFX contractnli --size base --mode full --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy \
    --holdout-qids-file data/pools/contractnli/holdout_qids.json --tag holdout$TAGSFX &
ft base-decoupled-contractnli-l11$TAGSFX contractnli $DEC --decouple-layers 11 --tag l11$TAGSFX &
wait
log "phase 2: decoupled l=7 and l=15, multi-head ECtHR"
ft base-decoupled-contractnli-l7$TAGSFX contractnli $DEC --decouple-layers 7 --tag l7$TAGSFX &
ft base-decoupled-contractnli-l15$TAGSFX contractnli $DEC --decouple-layers 15 --tag l15$TAGSFX &
mh ecthr --epochs 1 --select-points 10 &
wait
log "chain done"
