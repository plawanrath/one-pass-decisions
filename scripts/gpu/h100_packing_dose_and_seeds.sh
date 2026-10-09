#!/usr/bin/env bash
# GPU-host chain: the packing dose-response, the seed fill of the pooled table, and the timing controls.
# Phase A (the GPU otherwise idle): why engine-cold timed slower than engine-cold-fill (scripts/cold_path_diag.py); the
# main-text latency cells re-timed with every path serving each request once before timing (--warm-shapes, tag
# strength-warm); the full-attention fine-tune served joint-packed with no retraining (scripts/joint_packing.py).
# Phase B (training, four lanes at once, every checkpoint kept):
#   dose-response: ModernBERT base on contracts, block mask, the one-question-per-sequence arm's recipe with 2, 4, 8, 16
#     questions per sequence (five epochs, so equal question-examples seen; seeds 20260927 and 1), and 16 per sequence at
#     that arm's 5,951 optimizer steps;
#   pooled-table seeds: contracts base full seeds s1r/s2r and adapted seeds 1/2 from them; clauses adapted seed 1 (from
#     full seed 1), full s2r and adapted seed 2 from it; stories base adapted seeds 1/2; stories large full seeds 1/2 and
#     adapted from them; stories base mask-trained (from the RACE checkpoint, the full arm's 1,008 steps), three seeds;
#   ModernBERT-large mask-trained contract seeds 3 and 4;
#   optional tail: EuroBERT-610M mask-trained contract seeds 1 and 2, started only while DEADLINE_H (hours from the
#     chain's start, default 7.0; a 7.5 h session) leaves 2.2 h.
# REHEARSAL: EXTRA (appended to every fine-tune), TAGSFX, SKIP_TIMING=1, JEXTRA (joint_packing flags), INIT (where the
# uploaded starting checkpoints are; default: init/ on the host, after the chain moves them out of models/ so the model
# fetch returns only new checkpoints).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; TAGSFX=${TAGSFX:-}; SFX=${TAGSFX:+-$TAGSFX}; SKIP_TIMING=${SKIP_TIMING:-}; JEXTRA=${JEXTRA:-}
DEADLINE_H=${DEADLINE_H:-7.0}
T0=$(date +%s)
log() { echo "[dose-seeds] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
if [ -z "${INIT:-}" ]; then
  if [ -z "$TAGSFX" ] && ls -d models/_from_h100_* >/dev/null 2>&1; then mkdir -p init && mv models/_from_h100_* init/; fi
  INIT=init
fi
for t in contractnli quality; do [ -f data/pools/$t/train.jsonl ] || { log "missing data/pools/$t"; exit 1; }; done
if [ ! -f data/pools/cuad/train.jsonl ]; then  # the CUAD pools are rebuilt on the host and checked against the tracked manifest
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || log "pool lock FAILED"
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])" 2>/dev/null)
  if [ "$want" = "$got" ]; then rm data/pools/cuad/POOL.tracked.json; log "CUAD pools verified"; else log "CUAD manifest mismatch"; mv data/pools/cuad/POOL.tracked.json data/pools/cuad/POOL.json; fi
fi
for m in _from_h100_lr/modernbert-base-full-contractnli-lr2e5-bestacc _from_h100_seed/modernbert-base-full-cuad-s1 \
         _from_h100_round15/modernbert-base-full-quality-s1 _from_h100_round15/modernbert-base-full-quality-s2 \
         _from_h100_round15/modernbert-base-full-race _from_h100_round15/modernbert-large-full-race; do
  [ -d "$INIT/$m" ] || log "missing starting checkpoint $INIT/$m (runs that need it are skipped)"
done

ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), 'steps', d['train']['steps'], 's', d['train']['seconds'])" 2>/dev/null || echo FAILED)"
}

if [ -z "$SKIP_TIMING" ]; then
  log "phase A1: engine-cold against engine-cold-fill, interleaved, first and second pass"
  $PY scripts/cold_path_diag.py --device cuda 2>&1 | grep --line-buffered -E "\[diag\]|Error|Traceback"
  log "phase A2: main-text latency cells, every path warmed on every request shape (tag strength-warm)"
  lat() { $PY scripts/latency_sweep.py --device cuda --dtype bf16 --no-cudnn-sdp --stock --warm-shapes "$@" 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"; }
  TAG=warm
  for size in base large; do
    lat --size $size --task contractnli --n 50 --cells 2048:17 4096:17 --paths varlen-batched --out-tag strength-$TAG
    lat --size $size --task contractnli --n 50 --cells 2048:17 4096:17 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
    lat --size $size --task contractnli --n 30 --cells 4096:1 4096:2 4096:5 4096:10 --paths varlen-batched --out-tag strength-$TAG
    lat --size $size --task contractnli --n 30 --cells 4096:1 4096:2 4096:5 4096:10 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
  done
  lat --size base --task contractnli --n 20 --cells 4096:17 --paths one-seq compile-one-seq varlen-one-seq naive-batch --out-tag strength-$TAG
  [ -f data/pools/cuad/test.jsonl ] && lat --size base --task cuad --n 30 --cells 8192:41 --paths varlen-batched --out-tag strength-$TAG
  [ -f data/pools/cuad/test.jsonl ] && lat --size base --task cuad --n 30 --cells 8192:41 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
  [ -f data/pools/cuad/test.jsonl ] && lat --size base --task cuad --n 20 --cells 32768:41 --paths varlen-batched engine-windowed-cold engine-windowed-hit --engine-kernel split --out-tag strength-$TAG-windowed
  lat --size base --task quality --n 30 --cells 8192:17 --paths varlen-batched --out-tag strength-$TAG
  lat --size base --task quality --n 30 --cells 8192:17 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
fi
log "phase A3: full-attention weights served joint-packed, no retraining"
$PY scripts/joint_packing.py --served-full $INIT/_from_h100_lr/modernbert-base-full-contractnli-lr2e5-bestacc --served-result base-full-contractnli-lr2e5-bestacc $JEXTRA 2>&1 | grep --line-buffered -E "\[joint\]|Error|Traceback"

log "phase B: training lanes"
R="--size base --mode block --max-seq-tokens 5120 --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy"  # the unpacked arm's recipe
ADAPT="--mode block --epochs 2 --lr 1e-5 --max-seq-tokens 5120 --select-points 4 --select-metric last"
CADAPT="--mode block --epochs 2 --lr 1e-5 --max-seq-tokens 12288 --max-questions-per-seq 41 --max-seqs 1 --select-points 4 --select-metric last"
Q="--max-option-tokens 48 --tokens-per-batch 65536 --max-seqs 64"
AD="--mode block --epochs 2 --lr 1e-5 --max-option-tokens 48 --max-seq-tokens 12288 --max-questions-per-seq 32 --max-seqs 1 --select-points 4 --select-metric last"
QMASK="--size base --mode block --max-steps 1008 --lr 2e-5 --micro-tokens 16384 --max-seq-tokens 12288 --max-questions-per-seq 32 $Q --init-from $INIT/_from_h100_round15/modernbert-base-full-race"
qmask() {  # stories mask-trained seed ("" for the default seed)
  local s=$1
  ft base-block-quality-steps${s:+-s$s}$SFX quality $QMASK ${s:+--seed $s} --tag steps${s:+-s$s}$SFX
}

lane1() {  # clauses, stories base adapted, one stories mask-trained seed
  [ -d $INIT/_from_h100_seed/modernbert-base-full-cuad-s1 ] && ft base-block-cuad-adapt2-s1$SFX cuad --size base $CADAPT --seed 1 --init-from $INIT/_from_h100_seed/modernbert-base-full-cuad-s1 --tag adapt2-s1$SFX
  ft base-full-cuad-s2r$SFX cuad --size base --mode full --epochs 3 --tokens-per-batch 65536 --max-seqs 64 --micro-tokens 16384 --seed 2 --tag s2r$SFX
  [ -d models/modernbert-base-full-cuad-s2r$SFX ] && ft base-block-cuad-adapt2-s2$SFX cuad --size base $CADAPT --seed 2 --init-from models/modernbert-base-full-cuad-s2r$SFX --tag adapt2-s2$SFX
  for s in 1 2; do
    [ -d $INIT/_from_h100_round15/modernbert-base-full-quality-s$s ] && ft base-block-quality-adapt2-s$s$SFX quality --size base $AD --seed $s --init-from $INIT/_from_h100_round15/modernbert-base-full-quality-s$s --tag adapt2-s$s$SFX
  done
  qmask 2
}
lane2() {  # dose-response at 2 and 4 per sequence, then the large mask-trained seeds
  for k in 2 4; do
    ft base-block-contractnli-pack$k$SFX contractnli $R --max-questions-per-seq $k --tag pack$k$SFX
    ft base-block-contractnli-pack$k-s1$SFX contractnli $R --max-questions-per-seq $k --seed 1 --tag pack$k-s1$SFX
  done
  for s in 3 4; do
    ft large-block-contractnli-e30-s${s}r$SFX contractnli --size large --mode block --max-seq-tokens 5120 --max-steps 4616 --seed $s --tag e30-s${s}r$SFX
  done
}
lane3() {  # dose-response at 8 and 16 per sequence, stories large full and adapted, one stories mask-trained seed
  for k in 8 16; do
    ft base-block-contractnli-pack$k$SFX contractnli $R --max-questions-per-seq $k --tag pack$k$SFX
    ft base-block-contractnli-pack$k-s1$SFX contractnli $R --max-questions-per-seq $k --seed 1 --tag pack$k-s1$SFX
  done
  if [ -d $INIT/_from_h100_round15/modernbert-large-full-race ]; then
    for s in 1 2; do
      ft large-full-quality-s$s$SFX quality --size large --mode full --epochs 4 --lr 1e-5 --micro-tokens 8192 --select-points 8 --select-metric accuracy \
         --init-from $INIT/_from_h100_round15/modernbert-large-full-race $Q --seed $s --tag s$s$SFX
      [ -d models/modernbert-large-full-quality-s$s$SFX ] && ft large-block-quality-adapt2-s$s$SFX quality --size large $AD --seed $s --init-from models/modernbert-large-full-quality-s$s$SFX --tag adapt2-s$s$SFX
    done
  fi
  qmask 1
}
lane4() {  # the equal-steps cell, contracts base full and adapted seeds, one stories mask-trained seed
  ft base-block-contractnli-pack16-steps$SFX contractnli $R --max-questions-per-seq 16 --max-steps 5951 --tag pack16-steps$SFX
  for s in 1 2; do
    ft base-full-contractnli-lr2e5-bestacc-s${s}r$SFX contractnli --size base --mode full --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy --seed $s --tag lr2e5-bestacc-s${s}r$SFX
    [ -d models/modernbert-base-full-contractnli-lr2e5-bestacc-s${s}r$SFX ] && ft base-block-contractnli-adapt2-s$s$SFX contractnli --size base $ADAPT --seed $s \
        --init-from models/modernbert-base-full-contractnli-lr2e5-bestacc-s${s}r$SFX --tag adapt2-s$s$SFX
  done
  qmask ""
}
lane1 & lane2 & lane3 & lane4 &
wait
log "phase B done"
left=$(.venv/bin/python -c "print(int(($T0 + $DEADLINE_H * 3600 - $(date +%s)) / 60))")
if [ "$left" -ge 132 ]; then
  log "optional tail: EuroBERT-610M mask-trained seeds 1 and 2 ($left min left)"
  for s in 1 2; do
    ft eurobert-large-block-contractnli-steps-s$s$SFX contractnli --size eurobert-large --mode block --max-seq-tokens 5120 --micro-tokens 6144 --max-steps 4577 --seed $s --tag steps-s$s$SFX &
  done
  wait
else
  log "optional tail skipped ($left min left)"
fi
ls models/ | grep -v "^_from" | sed 's/^/[dose-seeds] model kept: /'
log "chain done"
