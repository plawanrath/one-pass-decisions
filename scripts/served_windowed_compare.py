"""Served masked beyond one window: the clause full-attention fine-tunes served under the four-window mask with no
training (scripts/served_windowed.py) against the same weights on one window (full attention, and served under the
one-window block mask) and against the windowed mask-trained model of the same seed where one exists; over all test
questions and over the questions whose first evidence starts beyond the first window's content, counted in each family's
own tokens (ModernBERT base and EuroBERT-610M tokenize differently, so each has its own subset).
Paired bootstrap clustered by contract (10,000 resamples).

    .venv/bin/python scripts/served_windowed_compare.py
Writes results/served_windowed/summary.json.
"""
from __future__ import annotations

import bisect

from onepass.analysis import paired_compare, read_scores
from onepass.cli import ENCODERS, RESULTS, pool_dir, state_tokens, write_json
from onepass.metrics import aurc
from onepass.model.encoder import load_tokenizer_adapter
from onepass.tasks.pools import load_pool

FT, SW = RESULTS / "finetune", RESULTS / "served_windowed"
SEEDS = (("base", "0", "base-full-cuad", "base-windowed-cuad-steps"), ("base", "1", "base-full-cuad-s1", "base-windowed-cuad-steps-s1"),
         ("eurobert-large", "0", "eurobert-large-full-cuad", None), ("eurobert-large", "1", "eurobert-large-full-cuad-s1", None))


def window_subsets(size: str) -> dict:
    """Test-pool indices of positive questions whose first evidence starts beyond / inside the first window's content
    tokens, in this family's tokenizer (as scripts/cuad_beyond_window.py counts them)."""
    recs = load_pool(pool_dir("cuad"), "test")
    tok, _ = load_tokenizer_adapter(*ENCODERS[size])
    limit, offsets, beyond, inside = state_tokens("cuad"), {}, [], []
    for i, r in enumerate(recs):
        c = r.meta.get("evidence_start_char")
        if c is None:
            continue
        if r.state_hash not in offsets:
            offsets[r.state_hash] = [o[0] for o in tok(r.state, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
        (beyond if bisect.bisect_right(offsets[r.state_hash], c) - 1 >= limit else inside).append(i)
    return {"all": None, "evidence_beyond_window": beyond, "evidence_inside_window": inside}


if __name__ == "__main__":
    out = {"seeds": []}
    cache = {}
    for size, seed, full, windowed in SEEDS:
        f = SW / full / "scores_test.jsonl"
        if not f.exists():
            continue
        subsets = cache.setdefault(size, window_subsets(size))
        served_w = read_scores(f)
        refs = {"one window, full attention": read_scores(FT / full / "scores_test.jsonl"),
                "one window, served masked": read_scores(FT / full / "scores_test_as_block.jsonl")}
        if windowed:
            refs["four windows, mask-trained"] = read_scores(FT / windowed / "scores_test.jsonl")
        row = {"size": size, "seed": seed, "accuracy": sum(r["pred"] == r["gold"] for r in served_w) / len(served_w),
               "aurc": aurc([r["conf"] for r in served_w], [int(r["pred"] != r["gold"]) for r in served_w]), "against": {}}
        for name, ref in refs.items():
            row["against"][name] = {}
            for sub, idx in subsets.items():
                a, b = (served_w, ref) if idx is None else ([served_w[i] for i in idx], [ref[i] for i in idx])
                c = paired_compare(a, b, cluster=True)
                row["against"][name][sub] = {"n": len(a), "accuracy_pp": c["accuracy_pp"], "aurc": c["aurc"]}
                print(f"[served-windowed] {size} seed {seed} minus {name} on {sub} (n={len(a)}): {c['accuracy_pp']['point']:+.1f} "
                      f"[{c['accuracy_pp']['ci_low']:+.1f}, {c['accuracy_pp']['ci_high']:+.1f}] pp; AURC {c['aurc']['point']:+.3f}")
        out["seeds"].append(row)
    write_json(SW / "summary.json", out)
