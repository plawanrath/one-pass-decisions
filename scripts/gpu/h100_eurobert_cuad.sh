#!/usr/bin/env bash
# GPU-host chain: the served-windowed result on a second encoder family. EuroBERT-610M fine-tuned with full attention on
# CUAD clauses at one 8,192-token window (seeds 1 and 0 one after the other, since two lanes side by side exceed 80 GB; lr 1e-5, EuroBERT-610M's tuned rate on contracts;
# two epochs; best of four checkpoints on dev accuracy), each then served under the four-window mask with no training
# (scripts/served_windowed.py, the ModernBERT windowed arm's configuration). Every checkpoint is kept (the session
# fetches models/). "only <seed>" runs one seed. REHEARSAL: positional args "[only <seed>] rehearse <extra finetune
# flags...>" (the provider runs --tasks under timeout, so no VAR= prefixes). The pod needs
# results/finetune/base-windowed-cuad-steps/summary.json (pass it with the provider's --upload).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=""; SFX=""; SIZE="eurobert-large"; SW=""; SEEDS="1 0"
if [ "${1:-}" = only ]; then SEEDS="$2"; shift 2; fi
if [ "${1:-}" = rehearse ]; then shift; EXTRA="$*"; SFX="-rh"; SIZE="eurobert-base"; SW="--max-eval 6 --max-state-tokens 1024 --window-tokens 256"; fi
log() { echo "[eurobert-cuad] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
log "phase 0: CUAD pools from the raw corpus, verified against the tracked manifest"
if [ ! -f data/pools/cuad/train.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch: $got != $want"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified ($got)"
fi
lane() {  # seed tag
  local seed=$1 tag=$2 name=$SIZE-full-cuad${2:+-$2}
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task cuad --size $SIZE --mode full --epochs 2 --lr 1e-5 \
    --tokens-per-batch 65536 --max-seqs 64 --micro-tokens 16384 --select-points 4 --select-metric accuracy --seed $seed --tag "$tag" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'one-window masked', round(s['test_as_block']['accuracy'],4), 'steps', d['train']['steps'], 's', d['train']['seconds'])" 2>/dev/null || echo FAILED)"
  [ -d models/$name ] && { [ -f results/served_windowed/$name/summary.json ] || $PY scripts/served_windowed.py --model models/$name --name $name $SW 2>&1 | grep --line-buffered -E "\[served-windowed\]|Error|Traceback"; }
}
[ -f results/finetune/base-windowed-cuad-steps/summary.json ] || { log "missing results/finetune/base-windowed-cuad-steps/summary.json (upload it)"; exit 1; }
for s in $SEEDS; do
  log "EuroBERT-610M full attention on one window, seed $s, then served under four windows"
  if [ "$s" = 0 ]; then lane 0 "${SFX#-}"; else lane $s "s$s$SFX"; fi
done
ls models/ | sed 's/^/[eurobert-cuad] model kept: /'
log "chain done"
