# data/

- `pools/<task>/POOL.json` — tracked manifests of the locked decision pools: spec (task, seed, sizes, source corpus),
  per-split row counts and sha256. The `*.jsonl` rows themselves are regenerated deterministically by
  `scripts/lock_pools.py` (routing, routing150, guardrail, contractnli, cuad, ecthr, quality, race) and `scripts/silver_labels.py` (tracegrade) and are
  verified against the manifest on load; they are not tracked because they are derived from public corpora.
- Corpora and licences: CLINC150 `clinc/clinc_oos` (CC BY 3.0); `deepset/prompt-injections` (Apache-2.0);
  `jackhhao/jailbreak-classification` (Apache-2.0); `openai/gsm8k` (MIT); ContractNLI (CC BY 4.0); CUAD v1 (CC BY 4.0); LexGLUE ECtHR `coastalcph/lex_glue` (CC BY 4.0);
  QuALITY v1.0.1 (CC BY 4.0; `QuALITY.v1.0.1.zip` from the authors' repository, sha256 in `onepass/tasks/quality.py`);
  RACE `ehovy/race` (research use; intermediate training only). No corpus is redistributed here.
