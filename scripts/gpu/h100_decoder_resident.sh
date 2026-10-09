#!/usr/bin/env bash
# GPU-host chain: the decoder's resident latency on the tight PyTorch path (the same cascade attention as the engine),
# beside the engine in one process with warm shapes, on contracts (base against SmolLM2-360M, large against
# SmolLM2-1.7B) and clauses (base against SmolLM2-360M); then the matched-accuracy load curve under Poisson arrivals.
# Needs data/cache/requests/{contractnli,cuad}.json and, for the agreement check, the 360M fine-tunes uploaded
# (models/_from_h100_flex/decoder-smol360-{contractnli,cuad}-lr1e4); the 1.7B model is the pinned pretrained one.
# REHEARSAL: DEV=cpu, REXTRA/LEXTRA (smaller settings for the two scripts), TAGSFX.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
DEV=${DEV:-cuda}; REXTRA=${REXTRA:-}; LEXTRA=${LEXTRA:-}; TAGSFX=${TAGSFX:-}; TAG=${TAGSFX:+--tag $TAGSFX}
log() { echo "[decoder-resident] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for t in contractnli cuad; do [ -f data/cache/requests/$t.json ] || { log "missing data/cache/requests/$t.json"; exit 1; }; done
# the pinned decoder tokenizers into the local cache first: the tests read them with local_files_only
$PY -c "from transformers import AutoTokenizer; from onepass.cli import DECODERS; [AutoTokenizer.from_pretrained(DECODERS[k][0], revision=DECODERS[k][1]) for k in ('smol360', 'smol1.7')]" 2>&1 | grep -E "Error|Traceback"
$PY -m pytest -q tests/test_decoder_cascade.py tests/test_decoder.py 2>&1 | tail -3
log "contracts 4096:17: engine and decoder, cold and resident"
$PY scripts/decoder_resident.py --device $DEV --pairs base:smol360 large:smol1.7 --n 50 --warmup 10 $REXTRA $TAG 2>&1 | grep --line-buffered -E "\[resident\]|Error|Traceback"
log "clauses 8192:41: base against SmolLM2-360M"
$PY scripts/decoder_resident.py --device $DEV --task cuad --cell 8192:41 --pairs base:smol360 --n 30 --warmup 10 $REXTRA $TAG 2>&1 | grep --line-buffered -E "\[resident\]|Error|Traceback"
log "load curve at matched accuracy"
$PY scripts/decoder_load.py --device $DEV --pairs large:smol1.7 base:smol360 --rates 5 10 20 30 40 60 --n 150 $LEXTRA $TAG 2>&1 | grep --line-buffered -E "\[load\]|Error|Traceback"
log "chain done"
