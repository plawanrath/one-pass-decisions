#!/usr/bin/env bash
# GPU-host chain: a non-legal long-document task and a clean joint-packing latency.
# Phase 0 (quiet GPU): the joint-packing arm re-timed with every engine built once (the earlier timing built an engine
# inside each timed call), and QuALITY-shaped requests (8,192-token stories, 17 four-option questions) through the
# unpadded baseline and the cascade engine. Phase 0b: the clause full-attention fine-tunes (seeds 0 and 1) served under the
# four-window mask with no training (scripts/served_windowed.py). Phase 1: RACE intermediate training (full attention, one epoch) for
# ModernBERT base and large, as QuALITY baselines do. Phase 2-3: QuALITY full-attention fine-tunes from the RACE
# checkpoints, base seeds 0/1/2 and large seed 0; each also scores its weights served under the block mask. Phase 4:
# brief mask adaptation (2 epochs, lr 1e-5, final checkpoint) of large seed 0 and base seed 0. Never more than two
# training jobs at a time. REHEARSAL: EXTRA (extra fine-tune arguments) and TAGSFX run the same chain on a tiny pool.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; TAGSFX=${TAGSFX:-}; SKIP_LAT=${SKIP_LAT:-}; SFX=${TAGSFX:+-$TAGSFX}
log() { echo "[round15] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for t in quality race contractnli cuad; do [ -f data/pools/$t/train.jsonl ] || { log "missing data/pools/$t (upload the locked pools)"; exit 1; }; done
ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];o=[k for k in s if k.startswith('test_as_')];print('dev', round(s['dev']['accuracy'],4), 'test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), *(f'{k} {s[k][\"accuracy\"]:.4f}' for k in o))" 2>/dev/null)"
}
if [ -z "$SKIP_LAT" ]; then
  log "phase 0: joint-packing re-time; QuALITY-shaped latency"
  $PY scripts/joint_packing.py --latency-only --n 50 2>&1 | grep --line-buffered -E "\[joint\]|Error|Traceback"
  lat() { $PY scripts/latency_sweep.py --device cuda --dtype bf16 --no-cudnn-sdp --stock "$@" 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"; }
  lat --size base --task quality --n 30 --cells 8192:17 --paths varlen-batched --out-tag strength-round15
  lat --size base --task quality --n 30 --cells 8192:17 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-round15
fi
if [ -z "$SKIP_LAT" ]; then
  log "phase 0b: clause full-attention weights served under the four-window mask (no training)"
  for m in models/_from_h100_cuad/modernbert-base-full-cuad:base-full-cuad models/_from_h100_seed/modernbert-base-full-cuad-s1:base-full-cuad-s1; do
    [ -f results/served_windowed/${m#*:}/summary.json ] || $PY scripts/served_windowed.py --model ${m%%:*} --name ${m#*:} 2>&1 | grep --line-buffered -E "\[served-windowed\]|Error|Traceback"
  done
fi
Q="--max-option-tokens 48 --tokens-per-batch 65536 --max-seqs 64"
log "phase 1: RACE intermediate training (base, large)"
ft base-full-race$SFX race --size base --mode full --epochs 1 --lr 2e-5 --max-option-tokens 48 --tokens-per-batch 32768 --max-seqs 64 --tag "$TAGSFX" &
ft large-full-race$SFX race --size large --mode full --epochs 1 --lr 1e-5 --max-option-tokens 48 --tokens-per-batch 32768 --max-seqs 64 --micro-tokens 16384 --tag "$TAGSFX" &
wait
RB=models/modernbert-base-full-race$SFX; RL=models/modernbert-large-full-race$SFX
[ -d $RB ] || { log "RACE base checkpoint missing"; exit 1; }
QB="--size base --mode full --epochs 4 --lr 2e-5 --micro-tokens 16384 --select-points 8 --select-metric accuracy --init-from $RB"
log "phase 2: QuALITY base seeds 0 and 1"
ft base-full-quality$SFX quality $QB $Q --tag "$TAGSFX" &
ft base-full-quality-s1$SFX quality $QB $Q --seed 1 --tag s1$SFX &
wait
log "phase 3: QuALITY base seed 2; large seed 0"
ft base-full-quality-s2$SFX quality $QB $Q --seed 2 --tag s2$SFX &
[ -d $RL ] && ft large-full-quality$SFX quality --size large --mode full --epochs 4 --lr 1e-5 --micro-tokens 8192 --select-points 8 --select-metric accuracy --init-from $RL $Q --tag "$TAGSFX" &
wait
log "phase 4: brief mask adaptation (large seed 0, base seed 0)"
AD="--mode block --epochs 2 --lr 1e-5 --max-option-tokens 48 --max-seq-tokens 12288 --max-questions-per-seq 32 --max-seqs 1 --select-points 4 --select-metric last"
[ -d models/modernbert-large-full-quality$SFX ] && ft large-block-quality-adapt2$SFX quality --size large $AD --init-from models/modernbert-large-full-quality$SFX --tag adapt2$SFX &
[ -d models/modernbert-base-full-quality$SFX ] && ft base-block-quality-adapt2$SFX quality --size base $AD --init-from models/modernbert-base-full-quality$SFX --tag adapt2$SFX &
wait
log "chain done"
