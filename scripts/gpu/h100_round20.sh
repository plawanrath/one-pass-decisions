#!/usr/bin/env bash
# GPU-host chain (training pod): (1) the mask-trained arm at its own learning rate with checkpoint selection, seeds 1
# and 2 (block mask, 16 questions per sequence, 5,951 steps, lr 3e-5, best of 10 checkpoints on dev accuracy:
# base-block-contractnli-bestacc's recipe), beside (2) EuroBERT-2.1B full attention, seed 1 at lr 5e-6 (its 1e-5 run did
# not converge), then adapted from it (2 epochs under the mask, lr 1e-5, final checkpoint). Every checkpoint is kept
# (the session fetches models/). REHEARSAL: positional args "rehearse <extra finetune flags...>" (the provider runs
# --tasks under timeout, so no VAR= prefixes).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=""; SFX=""; XL="eurobert-xl"
if [ "${1:-}" = rehearse ]; then shift; EXTRA="$*"; SFX="-rh"; XL="${XL_SIZE:-eurobert-base}"; fi
log() { echo "[round20] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/train.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
ft() {
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];o=s.get('test_as_block');print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), *(['served-masked', round(o['accuracy'],4)] if o else []), 'steps', d['train']['steps'], 's', d['train']['seconds'])" 2>/dev/null || echo FAILED)"
}
OWN="--size base --mode block --tokens-per-batch 16384 --max-seq-tokens 5120 --max-steps 5951 --lr 3e-5 --select-points 10 --select-metric accuracy"
FULL="--size $XL --mode full --epochs 4 --lr 5e-6 --micro-tokens 5120 --grad-checkpoint --select-points 5 --select-metric accuracy --save-dtype bf16"
ADAPT="--size $XL --mode block --epochs 2 --lr 1e-5 --max-seq-tokens 5120 --micro-tokens 5120 --grad-checkpoint --select-points 4 --select-metric last --save-dtype bf16"
log "phase 1: EuroBERT-2.1B seed 1 at lr 5e-6, with the two own-learning-rate mask-trained seeds beside it"
ft $XL-full-contractnli-lr5e6-s1$SFX $FULL --seed 1 --tag lr5e6-s1$SFX &
for s in 1 2; do ft base-block-contractnli-bestacc-s$s$SFX $OWN --seed $s --tag bestacc-s$s$SFX & done
wait
log "phase 2: adapted from the 5e-6 run"
M=models/$XL-full-contractnli-lr5e6-s1$SFX
[ -d $M ] && ft $XL-block-contractnli-adapt2-lr5e6-s1$SFX $ADAPT --seed 1 --init-from $M --tag adapt2-lr5e6-s1$SFX
ls models/ | sed 's/^/[round20] model kept: /'
log "chain done"
