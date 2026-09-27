# RUNBOOK

Every number in the paper has a command here. Results land in `results/<experiment>/` as JSON summaries plus
per-row JSONL; fine-tuned weights land in `models/` (not tracked). Hosts are labelled by chip family and OS only.

## 0. One-time setup

```bash
bash scripts/env/setup.sh                       # Python 3.11 venv from the frozen lock for this platform, plus `pip install -e .`
.venv/bin/python scripts/env/pin_models.py --from-lock   # downloads the pinned model revisions into the Hugging Face cache
```

`scripts/env/requirements.txt` is the loose, hand-edited set. `scripts/env/requirements.lock.<platform>.txt` is the
frozen set from a real install on that platform (`bash scripts/env/lock.sh` regenerates it). On a platform without a
lock, `setup.sh` resolves from the loose set and writes a new lock. `scripts/env/models.lock.txt` pins every upstream
model to a commit hash.

## 1. Tests

```bash
.venv/bin/python -m pytest -q -m "not slow and not gpu"     # fast subset (CI, ~5 s): layout, masks, exact-reuse invariants, engine, gate, pools, traces
.venv/bin/python -m pytest -q                               # everything, including the ONNX export round trip
```

The invariance tests in `tests/test_model.py` are the mechanism's correctness argument: the state's activations do not
change when questions are added, a question's scores do not change with the questions packed beside it, and the
cache-hit path reproduces the cold packed pass.

## 2. Data

- `data/pools/<task>/{train,dev,cal,test}.jsonl` + `POOL.json` — locked decision pools (manifest with n, seed, sha256).
  Built once by `scripts/lock_pools.py`; loaders verify the hash. Sources and licences:
  - routing: CLINC150 (`clinc/clinc_oos`, config `plus`), CC BY 3.0; official splits kept.
  - guardrail: `deepset/prompt-injections` (Apache-2.0) and `jackhhao/jailbreak-classification` (Apache-2.0); group split by turn.
  - tracegrade: trajectories from a code agent over `openai/gsm8k` (MIT), labels programmatic or System 2 self-consistency.
- `results/traces/` — request traces (guardrail pipeline with simulated Poisson arrivals; agent harness with measured step times).

## 3. Pools, traces, fine-tunes, K = 1 timings

```bash
.venv/bin/python scripts/lock_pools.py --task routing
.venv/bin/python scripts/lock_pools.py --task routing150          # the 150-option layout
.venv/bin/python scripts/lock_pools.py --task guardrail
.venv/bin/python scripts/traces_guardrail.py --rate 20
.venv/bin/python scripts/traces_agent.py --n 300 --max-steps 8 2>&1 | tee logs/traces_agent.log     # hours; resumable
.venv/bin/python scripts/silver_labels.py --samples 5 2>&1 | tee logs/silver.log                    # hours; resumable; locks the tracegrade pools
.venv/bin/python scripts/finetune.py --size base --mode full  --task routing 2>&1 | tee logs/ft_base_full_routing.log
.venv/bin/python scripts/finetune.py --size base --mode block --task routing 2>&1 | tee logs/ft_base_block_routing.log
.venv/bin/python scripts/baselines_k1.py --size base --task routing --threads 8
```

Long runs: `.venv/bin/python scripts/... 2>&1 | tee logs/<name>.log`. Fine-tunes checkpoint every 200 steps and
resume from `models/<name>/ckpt.pt`; the agent harness and the System 2 labeller skip finished items on restart.
