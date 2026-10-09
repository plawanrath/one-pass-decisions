#!/usr/bin/env bash
# GPU-host chain: the baselines a fixed question set invites, and questions the model never saw.
# Part 1: the multi-head baseline (one state-only encode per document, one head per question; scripts/multihead_finetune.py)
# on contracts, clauses and ECtHR. Part 2: the held-out-question arm on contracts (ModernBERT base, full attention, lr 2e-5,
# 5 epochs, best dev accuracy over the training questions' dev records; data/pools/contractnli/holdout_qids.json), whose
# other-layout scores give the served-masked engine on the held-out questions; the multi-head baseline has no head for
# them. Never more than two training jobs on the GPU at a time (three large jobs ran out of memory before).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[baselines] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
lock() {  # task: lock the pools on this host and check the manifest
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
  [ -f results/multihead/base-$t/summary.json ] || $PY scripts/multihead_finetune.py --task $t --size base "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/multihead_$t.log
  log "done multihead $t: $(.venv/bin/python -c "import json;d=json.load(open('results/multihead/base-$t/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4), 'ms/doc', d['latency_ms_per_document'])" 2>/dev/null)"
}
ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'held-out', round(s.get('test_heldout',{}).get('accuracy',float('nan')),4), 'held-out served-masked', round(s.get('test_heldout_as_block',{}).get('accuracy',float('nan')),4))" 2>/dev/null)"
}
lock cuad
lock ecthr
log "part 1: multi-head baseline (contracts beside the held-out-question arm)"
mh contractnli --epochs 5 --select-points 10 &
ft base-full-contractnli-holdout contractnli --size base --mode full --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy \
    --holdout-qids-file data/pools/contractnli/holdout_qids.json --tag holdout &
wait
log "part 2: multi-head baseline on clauses and ECtHR"
mh cuad --epochs 3 --tokens-per-batch 65536 --max-docs 8 --select-points 5 &
mh ecthr --epochs 1 --select-points 10 &
wait
# --- hook: the DeFormer-style decoupled arm (another change adds `--mode decoupled --decouple-layers N` to
# scripts/finetune.py); add its runs here, two at a time, e.g.
#   ft base-decoupled-contractnli-l6 contractnli --size base --mode decoupled --decouple-layers 6 --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy --tag l6 &
log "chain done"
