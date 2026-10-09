#!/usr/bin/env bash
# GPU-host chain: the two ADR-0011 runs that ran out of GPU memory when three jobs shared the GPU (ModernBERT-large seed 2,
# EuroBERT-610M seed 1): their full-attention arms retrained two at a time, then adapted two at a time, with the same
# commands as scripts/gpu/h100_adapt.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[adapt] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
}
ADAPT="--mode block --epochs 2 --lr 1e-5 --max-seq-tokens 5120 --select-points 4 --select-metric last"
log "phase 1: full arms retrained (two at a time)"
ft large-full-contractnli-s2r contractnli --size large --mode full --epochs 4 --seed 2 --tag s2r &
ft eurobert-large-full-contractnli-lr1e5-s1r contractnli --size eurobert-large --mode full --epochs 4 --lr 1e-5 --micro-tokens 8192 --seed 1 --tag lr1e5-s1r &
wait
log "phase 2: adapted (two at a time)"
[ -d models/modernbert-large-full-contractnli-s2r ] && ft large-block-contractnli-adapt2-s2 contractnli --size large $ADAPT --seed 2 --init-from models/modernbert-large-full-contractnli-s2r --tag adapt2-s2 &
[ -d models/eurobert-large-full-contractnli-lr1e5-s1r ] && ft eurobert-large-block-contractnli-adapt2-s1 contractnli --size eurobert-large $ADAPT --micro-tokens 6144 --seed 1 --init-from models/eurobert-large-full-contractnli-lr1e5-s1r --tag adapt2-s1 &
wait
log "chain done"
