#!/usr/bin/env bash
# GPU-host chain: the round-13 jobs that failed — the MICE-style arm at l = 7 and 11 (its step count comes from the
# uploaded base-full-contractnli summary) and the clause multi-head sweep, two jobs at a time (four 8k-token jobs ran
# out of memory). Same commands as scripts/gpu/h100_round13.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; MHEXTRA=${MHEXTRA:-}; TAGSFX=${TAGSFX:-}
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
mh() {
  local t=$1 tag=$2$TAGSFX; shift 2
  [ -f results/multihead/base-$t-$tag/summary.json ] || $PY scripts/multihead_finetune.py --task $t --size base --tag $tag "$@" $MHEXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/multihead_$t-$tag.log
  log "done multihead $t $tag: $(.venv/bin/python -c "import json;d=json.load(open('results/multihead/base-$t-$tag/summary.json'));s=d['splits'];print('dev', round(s['dev']['accuracy'],4), 'test', round(s['test']['accuracy'],4))" 2>/dev/null)"
}
ft() {
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('dev', round(s['dev']['accuracy'],4), 'test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4))" 2>/dev/null)"
}
[ -f results/finetune/base-full-contractnli/summary.json ] || { log "missing base-full-contractnli summary (step count)"; exit 1; }
lock cuad
log "phase 1: MICE-style l=7 and l=11 (two jobs)"
for l in 7 11; do
  ft base-mice-contractnli-l$l$TAGSFX contractnli --size base --mode mice --decouple-layers $l --max-seq-tokens 5120 --match-steps-of base-full-contractnli --tag l$l$TAGSFX &
done
wait
log "phase 2: clause multi-head sweep (two at a time)"
for lr in 2e-5 5e-5; do
  for pool in cls mean; do mh cuad lr$lr-$pool --epochs 3 --tokens-per-batch 32768 --max-docs 4 --select-points 5 --lr $lr --pool $pool & done
  wait
done
log "chain done"
