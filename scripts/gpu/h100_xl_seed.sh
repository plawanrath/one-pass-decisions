#!/usr/bin/env bash
# GPU-host chain (training pod): one more EuroBERT-2.1B seed at its tuned learning rate on contracts. Full attention at
# lr 5e-6 (best of five checkpoints on dev accuracy, eurobert-xl-full-contractnli-lr5e6-s1's recipe), scored also served
# under the block mask, then adapted from it (2 epochs under the mask, lr 1e-5, final checkpoint). Every checkpoint is kept
# (the session fetches models/). Usage: h100_xl_seed.sh <seed> [rehearse <extra finetune flags...>] (the provider runs
# --tasks under timeout, so no VAR= prefixes).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
SEED=${1:?seed}; shift
EXTRA=""; SFX=""; XL="eurobert-xl"
if [ "${1:-}" = rehearse ]; then shift; EXTRA="$*"; SFX="-rh"; XL="eurobert-base"; fi
log() { echo "[xl-seed] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/train.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
ft() {
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];o=s.get('test_as_block');print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), *(['served-masked', round(o['accuracy'],4)] if o else []), 'steps', d['train']['steps'], 's', d['train']['seconds'])" 2>/dev/null || echo FAILED)"
}
FULL="--size $XL --mode full --epochs 4 --lr 5e-6 --micro-tokens 5120 --grad-checkpoint --select-points 5 --select-metric accuracy --save-dtype bf16"
ADAPT="--size $XL --mode block --epochs 2 --lr 1e-5 --max-seq-tokens 5120 --micro-tokens 5120 --grad-checkpoint --select-points 4 --select-metric last --save-dtype bf16"
log "phase 1: EuroBERT-2.1B full attention, seed $SEED, lr 5e-6"
ft $XL-full-contractnli-lr5e6-s$SEED$SFX $FULL --seed $SEED --tag lr5e6-s$SEED$SFX
log "phase 2: adapted from it"
M=models/$XL-full-contractnli-lr5e6-s$SEED$SFX
[ -d $M ] && ft $XL-block-contractnli-adapt2-lr5e6-s$SEED$SFX $ADAPT --seed $SEED --init-from $M --tag adapt2-lr5e6-s$SEED$SFX
ls models/ | sed 's/^/[xl-seed] model kept: /'
log "chain done"
