#!/usr/bin/env bash
# Local queue on the workstation. Every step skips when its artifact exists, so re-running resumes. Order:
#   wait for the agent capture -> silver labels + tracegrade pools -> fine-tunes (guardrail, routing 4-epoch, tracegrade)
#   -> per-path scores (3 tasks) -> G1 comparison -> K sweeps (routing, guardrail, tracegrade) -> thresholds.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
log() { echo "[queue] $(date '+%F %T') $*"; }
ft() {  # ft <task> <mode> <epochs> <tag>
  local task=$1 mode=$2 ep=$3 tag=$4
  local name="base-$mode-$task${tag:+-$tag}"
  if [ ! -f "results/finetune/$name/summary.json" ]; then
    log "finetune $name ($ep epochs)"
    $PY scripts/finetune.py --size base --mode $mode --task $task --epochs $ep ${tag:+--tag $tag} 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/ft_$name.log
  fi
}
# the second capture batch (problems 300-999) runs beside these fine-tunes; its step timings are not used for the
# arrival statistics (the first 300 trajectories ran uncontended and are the arrival sample)
ft guardrail full 1 ""
ft guardrail block 1 ""
ft routing block 4 e4
ft routing full 4 e4
while pgrep -f "traces_agent.py" > /dev/null; do sleep 60; done
log "agent capture finished"
if [ ! -f data/pools/tracegrade/POOL.json ]; then
  log "silver labels + tracegrade pools"
  $PY scripts/silver_labels.py --samples 5 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/silver.log
fi
ft tracegrade full 2 ""
ft tracegrade block 2 ""
for task in routing guardrail tracegrade; do
  if [ -d "models/modernbert-base-block-$task" ] && [ ! -f "results/path_scores/base-$task/meta.json" ]; then
    log "pathscores $task"
    $PY scripts/path_scores.py --size base --task $task --paths engine-hit ort-fp32 ort-int8 --threads 8 2>&1 | tee -a logs/pathscores_$task.log
  fi
done
log "G1 comparison"
$PY scripts/mask_price.py --size base --tasks routing guardrail tracegrade 2>&1 | tee -a logs/g1_mask.log
for task in routing guardrail tracegrade; do
  if [ -d "models/modernbert-base-block-$task" ] && [ ! -f "results/latency/arm64-darwin-base-$task-t8/summary.json" ]; then
    log "characterize $task 8 threads"
    $PY scripts/latency_sweep.py --size base --task $task --threads 8 --K 1 5 20 50 --S 512 2048 --paths one-seq naive-batch engine-cold engine-hit ort-one-seq ort-int8-one-seq ort-packed prefill gliner 2>&1 | tee -a logs/characterize_$task.log
  fi
done
for task in routing guardrail tracegrade; do
  if [ -f "results/path_scores/base-$task/meta.json" ]; then
    log "thresholds $task"
    $PY scripts/risk_paths.py --size base --task $task 2>&1 | tee -a logs/thresholds_$task.log
  fi
done
log "queue done"
