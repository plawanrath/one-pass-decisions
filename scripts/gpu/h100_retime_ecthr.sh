#!/usr/bin/env bash
# GPU-host chain: the main-text re-time (scripts/gpu/h100_retime.sh, GPU exclusive) and then the ECtHR mask-trained arm
# at the full arm's optimizer steps (the full arm trains on the workstation; its summary.json, which carries the step
# count, is copied to this host when it finishes). The ECtHR pools are locked and verified before the re-time so that no
# download competes with the timed cells.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[retime-ecthr] $(date '+%F %T') $*"; }
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
if [ ! -f data/pools/ecthr/test.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/ecthr/POOL.json'))['manifest_sha256'])")
  mv data/pools/ecthr/POOL.json data/pools/ecthr/POOL.tracked.json
  $PY scripts/lock_pools.py --task ecthr || { log "ECtHR pool lock FAILED"; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/ecthr/POOL.json'))['manifest_sha256'])" 2>/dev/null)
  if [ "$want" = "$got" ]; then rm data/pools/ecthr/POOL.tracked.json; log "ECtHR pools verified"; else log "ECtHR manifest mismatch: $got != $want"; mv data/pools/ecthr/POOL.tracked.json data/pools/ecthr/POOL.json; fi
fi
log "part 1: re-time"
bash scripts/gpu/h100_retime.sh
log "part 2: ECtHR mask-trained arm (waits up to 60 min for the full arm's summary)"
for _ in $(seq 60); do [ -f results/finetune/base-full-ecthr/summary.json ] && break; sleep 60; done
if [ -f results/finetune/base-full-ecthr/summary.json ] && [ -f data/pools/ecthr/test.jsonl ]; then
  $PY scripts/finetune.py --task ecthr --size base --mode block --max-seq-tokens 5120 --match-steps-of base-full-ecthr --tag steps 2>&1 \
      | grep --line-buffered -v "it/s\]" > logs/ft_base-block-ecthr-steps.log
  log "done base-block-ecthr-steps: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/base-block-ecthr-steps/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
else
  log "ECtHR arm skipped (no full-arm summary or no pools)"
fi
log "chain done"
