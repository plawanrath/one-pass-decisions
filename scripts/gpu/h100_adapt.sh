#!/usr/bin/env bash
# GPU-host chain: brief mask adaptation (ADR-0011, E21). A full-attention fine-tune is continued under the block mask for
# 2 epochs at lr 1e-5 (the mask-trained arm's packing), final checkpoint kept, dev curve recorded. Phase 0: base
# rehearsals (contracts, clauses). Phase 1: ModernBERT-large full arms of seeds 1 and 2 retrained (no weights were kept)
# beside the adaptation of seed 0. Phase 2: adaptation of seeds 1 and 2. Phase 3: EuroBERT-610M, three seeds retrained
# and adapted. REHEARSAL: EXTRA (extra fine-tune arguments) and TAGSFX (a tag suffix) run the same chain on a tiny pool.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}
TAGSFX=${TAGSFX:-}
log() { echo "[adapt] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ft() {  # result-name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
}
ADAPT="--mode block --epochs 2 --lr 1e-5 --max-seq-tokens 5120 --select-points 4 --select-metric last"
if [ ! -f data/pools/cuad/test.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])" 2>/dev/null)
  if [ "$want" = "$got" ]; then rm data/pools/cuad/POOL.tracked.json; log "CUAD pools verified"; else log "CUAD manifest mismatch"; mv data/pools/cuad/POOL.tracked.json data/pools/cuad/POOL.json; fi
fi
log "phase 0: base rehearsals"
ft base-block-contractnli-adapt2$TAGSFX contractnli --size base $ADAPT --init-from models/_from_h100_lr/modernbert-base-full-contractnli-lr2e5-bestacc --tag adapt2$TAGSFX &
[ -f data/pools/cuad/test.jsonl ] && ft base-block-cuad-adapt2$TAGSFX cuad --size base --mode block --epochs 2 --lr 1e-5 --max-seq-tokens 12288 --max-questions-per-seq 41 --max-seqs 1 \
    --select-points 4 --select-metric last --init-from models/_from_h100_cuad/modernbert-base-full-cuad --tag adapt2$TAGSFX &
wait
log "phase 1: ModernBERT-large seeds 1 and 2 retrained; seed 0 adapted"
ft large-full-contractnli-s1r$TAGSFX contractnli --size large --mode full --epochs 4 --seed 1 --tag s1r$TAGSFX &
ft large-full-contractnli-s2r$TAGSFX contractnli --size large --mode full --epochs 4 --seed 2 --tag s2r$TAGSFX &
ft large-block-contractnli-adapt2$TAGSFX contractnli --size large $ADAPT --init-from models/_from_h100_cnli/modernbert-large-full-contractnli --tag adapt2$TAGSFX &
wait
log "phase 2: ModernBERT-large seeds 1 and 2 adapted"
for s in 1 2; do
  [ -d models/modernbert-large-full-contractnli-s${s}r$TAGSFX ] && ft large-block-contractnli-adapt2-s$s$TAGSFX contractnli --size large $ADAPT --seed $s \
      --init-from models/modernbert-large-full-contractnli-s${s}r$TAGSFX --tag adapt2-s$s$TAGSFX &
done
wait
log "phase 3: EuroBERT-610M, three seeds retrained, then adapted"
ft eurobert-large-full-contractnli-lr1e5-r$TAGSFX contractnli --size eurobert-large --mode full --epochs 4 --lr 1e-5 --micro-tokens 8192 --tag lr1e5-r$TAGSFX &
for s in 1 2; do
  ft eurobert-large-full-contractnli-lr1e5-s${s}r$TAGSFX contractnli --size eurobert-large --mode full --epochs 4 --lr 1e-5 --micro-tokens 8192 --seed $s --tag lr1e5-s${s}r$TAGSFX &
done
wait
ebinit() { [ "$1" = 0 ] && echo models/eurobert-large-full-contractnli-lr1e5-r$TAGSFX || echo models/eurobert-large-full-contractnli-lr1e5-s${1}r$TAGSFX; }
for s in 0 1 2; do
  sfx=$([ "$s" = 0 ] && echo "" || echo "-s$s"); seedarg=$([ "$s" = 0 ] && echo "" || echo "--seed $s")
  [ -d "$(ebinit $s)" ] && ft eurobert-large-block-contractnli-adapt2$sfx$TAGSFX contractnli --size eurobert-large $ADAPT --micro-tokens 6144 $seedarg \
      --init-from "$(ebinit $s)" --tag adapt2$sfx$TAGSFX &
done
wait
log "chain done"
