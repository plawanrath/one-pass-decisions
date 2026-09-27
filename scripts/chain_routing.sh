#!/usr/bin/env bash
# Routing chain: fine-tunes (ModernBERT-base, full then block, 2 epochs), then the K=1 timing cells. Resumable: rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="/usr/bin/env .venv/bin/python -u"
for mode in full block; do
  if [ ! -f "results/finetune/base-$mode-routing/summary.json" ]; then
    .venv/bin/python -u scripts/finetune.py --size base --mode $mode --task routing --epochs 2 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/ft_base_${mode}_routing.log
  fi
done
.venv/bin/python -u scripts/baselines_k1.py --size base --task routing --threads 8 2>&1 | tee -a logs/baselines_k1_routing.log
echo "[chain] done $(date)"
