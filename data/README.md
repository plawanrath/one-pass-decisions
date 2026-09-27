# data/

- `pools/<task>/POOL.json` — tracked manifests of the locked decision pools: spec (task, seed, sizes, source corpus),
  per-split row counts and sha256. The `*.jsonl` rows themselves are regenerated deterministically by
  `scripts/w01_lock_pools.py` (routing, routing150, guardrail) and `scripts/w01_silver_labels.py` (tracegrade) and are
  verified against the manifest on load; they are not tracked because they are derived from public corpora.
- Corpora and licences: CLINC150 `clinc/clinc_oos` (CC BY 3.0); `deepset/prompt-injections` (Apache-2.0);
  `jackhhao/jailbreak-classification` (Apache-2.0); `openai/gsm8k` (MIT). No corpus is redistributed here.
