#!/usr/bin/env bash
# GPU-host chain: the training-length control for the long-document ranking result. The full-attention arm is trained
# for as many epochs as the mask-trained arm ran at equal optimizer steps (contracts 38.673, clauses 14.064), recording
# the dev curve at evenly spaced points and keeping the final checkpoint (as the mask-trained arm did); every run also
# scores its weights under the block mask. Usage: h100_epoch_control.sh contracts|clauses|all (all: one GPU, the clause
# run first, then each contract seed once enough GPU memory is free)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
WHICH=${1:?contracts, clauses or all}
log() { echo "[epochs] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ft() {  # name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
}
if [ "$WHICH" != contracts ] && [ ! -f data/pools/cuad/test.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified"
fi
log "phase 1: full arm at the mask-trained arm's epochs ($WHICH)"
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
contracts_seed() {  # seed (0 = the default training seed)
  local s=$1 sfx="" seedarg=""
  [ "$s" = 0 ] || { sfx="-s$s"; seedarg="--seed $s"; }
  ft base-full-contractnli-lr2e5-e39$sfx contractnli --size base --mode full --epochs 38.673 --lr 2e-5 --select-points 8 --select-metric last $seedarg --tag lr2e5-e39$sfx
}
clauses_run() {
  ft base-full-cuad-e14 cuad --size base --mode full --epochs 14.064 --tokens-per-batch 65536 --max-seqs 64 --micro-tokens 16384 --select-points 5 --select-metric last --tag e14
}
case "$WHICH" in
  contracts) for s in 0 1 2; do contracts_seed $s & done ;;
  clauses) clauses_run & ;;
  all)
    clauses_run &
    for s in 0 1 2; do
      sleep 300  # let the previous run reach its peak memory before measuring what is free
      until [ "$(free_mib)" -ge 24000 ]; do sleep 120; done
      log "starting contracts seed $s (free $(free_mib) MiB)"
      contracts_seed $s &
    done ;;
esac
wait
log "chain done"
