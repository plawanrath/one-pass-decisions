#!/usr/bin/env bash
# GPU-host chain for the round-13 review: (A) the multi-head baseline swept (contracts: lr 1e-5, 2e-5, 5e-5, 1e-4 x [CLS] or
# mean pooling; clauses: lr 2e-5, 5e-5 x both poolings), selection across runs on development accuracy only; (B) the
# DeFormer-style decoupled arm at l = 11 (lost to memory before) and the MICE-style arm (block mask from layer l up,
# question-to-state attention removed below) at l = 7 and 11, at the mask-trained arm's optimizer steps. Base-size jobs,
# at most four at a time. REHEARSAL: EXTRA / MHEXTRA / TAGSFX as in h100_round12.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; MHEXTRA=${MHEXTRA:-}; TAGSFX=${TAGSFX:-}; MICESTEPS=${MICESTEPS:---match-steps-of base-full-contractnli}
log() { echo "[round13] $(date '+%F %T') $*"; }
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
mh() {  # task tag args...
  local t=$1 tag=$2$TAGSFX; shift 2
  [ -f results/multihead/base-$t-$tag/summary.json ] || $PY scripts/multihead_finetune.py --task $t --size base --tag $tag "$@" $MHEXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/multihead_$t-$tag.log
  log "done multihead $t $tag: $(.venv/bin/python -c "import json;d=json.load(open('results/multihead/base-$t-$tag/summary.json'));s=d['splits'];print('dev', round(s['dev']['accuracy'],4), 'test', round(s['test']['accuracy'],4))" 2>/dev/null)"
}
ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('dev', round(s['dev']['accuracy'],4), 'test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4))" 2>/dev/null)"
}
lock cuad
log "phase A1: multi-head sweep, contracts"
for lr in 1e-5 2e-5 5e-5 1e-4; do
  mh contractnli lr$lr-cls --epochs 5 --select-points 10 --lr $lr --pool cls &
  mh contractnli lr$lr-mean --epochs 5 --select-points 10 --lr $lr --pool mean &
  if [ "$lr" = 2e-5 ] || [ "$lr" = 1e-4 ]; then wait; fi
done
wait
log "phase A2: multi-head sweep, clauses"
for lr in 2e-5 5e-5; do
  for pool in cls mean; do mh cuad lr$lr-$pool --epochs 3 --tokens-per-batch 32768 --max-docs 4 --select-points 5 --lr $lr --pool $pool & done
done
wait
log "phase B: decoupled l=11; MICE-style l=7 and l=11"
ft base-decoupled-contractnli-l11$TAGSFX contractnli --size base --mode decoupled --decouple-layers 11 --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy --tag l11$TAGSFX &
for l in 7 11; do
  ft base-mice-contractnli-l$l$TAGSFX contractnli --size base --mode mice --decouple-layers $l --max-seq-tokens 5120 $MICESTEPS --tag l$l$TAGSFX &
done
wait
log "chain done"
