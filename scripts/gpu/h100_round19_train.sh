#!/usr/bin/env bash
# GPU-host chain (training pod): EuroBERT-2.1B on contracts at matched size against SmolLM3-3B, and the mask-trained arm
# under the full arm's exact protocol for two more seeds.
#   EuroBERT-2.1B full attention, seeds 20260927 and 1: the EuroBERT-610M recipe (4 epochs, lr 1e-5, 16,384-token
#     batches), one sequence per micro-batch, gradient checkpointing (same gradients, about half the memory), best
#     checkpoint on dev accuracy like the full arms; each also scored served masked (the full runs' *_as_block scores);
#     backbone weights stored in bf16 (4 GB instead of 8); adapted from each: 2 epochs under the mask, lr 1e-5, final
#     checkpoint.
#   Symmetric protocol, ModernBERT base: block mask, 16 questions per sequence, the full arm's 5,951 steps, lr 2e-5,
#     best checkpoint on dev accuracy (base-block-contractnli-pack16-steps's recipe), seeds 1 and 2, beside the first
#     2.1B run (about 70 GB together).
# Every checkpoint is kept (the session fetches models/). REHEARSAL: positional args "rehearse <extra finetune flags...>"
# (the provider runs --tasks under timeout, so no VAR= prefixes), e.g.
#   bash scripts/gpu/h100_round19_train.sh rehearse --max-train 24 --max-eval 12 --max-state-tokens 256 --max-steps 2 --device cpu
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=""; SFX=""; XL="eurobert-xl"
if [ "${1:-}" = rehearse ]; then shift; EXTRA="$*"; SFX="-rh"; XL="${XL_SIZE:-eurobert-base}"; fi  # rehearsal: a small EuroBERT stands in
log() { echo "[round19-train] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/train.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
ft() {
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];o=s.get('test_as_block');print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), *(['served-masked', round(o['accuracy'],4)] if o else []), 'steps', d['train']['steps'], 's', d['train']['seconds'])" 2>/dev/null || echo FAILED)"
}
FULL="--size $XL --mode full --epochs 4 --lr 1e-5 --micro-tokens 5120 --grad-checkpoint --select-points 5 --select-metric accuracy --save-dtype bf16"
ADAPT="--size $XL --mode block --epochs 2 --lr 1e-5 --max-seq-tokens 5120 --micro-tokens 5120 --grad-checkpoint --select-points 4 --select-metric last --save-dtype bf16"
SYM="--size base --mode block --max-seq-tokens 5120 --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy --max-questions-per-seq 16 --max-steps 5951"
log "phase 1: EuroBERT-2.1B full seed 20260927, with the two symmetric-protocol seeds beside it"
ft $XL-full-contractnli-lr1e5$SFX $FULL --tag lr1e5$SFX &
for s in 1 2; do ft base-block-contractnli-pack16-steps-s$s$SFX $SYM --seed $s --tag pack16-steps-s$s$SFX & done
wait
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null | tail -1
log "phase 2: adapted from seed 20260927; full seed 1; adapted from seed 1 (one at a time: each about 38 GB)"
M=models/$XL-full-contractnli-lr1e5$SFX
[ -d $M ] && ft $XL-block-contractnli-adapt2$SFX $ADAPT --init-from $M --tag adapt2$SFX
ft $XL-full-contractnli-lr1e5-s1$SFX $FULL --seed 1 --tag lr1e5-s1$SFX
M=models/$XL-full-contractnli-lr1e5-s1$SFX
[ -d $M ] && ft $XL-block-contractnli-adapt2-s1$SFX $ADAPT --seed 1 --init-from $M --tag adapt2-s1$SFX
ls models/ | sed 's/^/[round19-train] model kept: /'
log "chain done"
