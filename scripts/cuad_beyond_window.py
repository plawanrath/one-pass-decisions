"""E4 (ADR-0005): does the windowed state read what one window cannot? Paired comparison of the windowed block model
(four windows, 32,768 layout tokens) with the one-window block model (8,190 state tokens) on CUAD's test pool: over
all questions, and over the questions whose first marked evidence starts beyond the first window's content.

    .venv/bin/python scripts/cuad_beyond_window.py --size base [--tag steps]
Reads results/finetune/<size>-{windowed,block}-cuad[-<tag>]/scores_test.jsonl; writes
results/mask_price/<size>[-<tag>]-windowed-vs-block/beyond_window.json.
"""
from __future__ import annotations

import argparse
import bisect

from onepass.analysis import paired_compare, read_scores
from onepass.cli import ENCODERS, RESULTS, pool_dir, state_tokens, write_json
from onepass.model.encoder import load_tokenizer_adapter
from onepass.tasks.pools import load_pool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = f"-{a.tag}" if a.tag else ""
    win = read_scores(RESULTS / "finetune" / f"{a.size}-windowed-cuad{sfx}" / "scores_test.jsonl")
    one = read_scores(RESULTS / "finetune" / f"{a.size}-block-cuad{sfx}" / "scores_test.jsonl")
    if len(win) != len(one):
        raise SystemExit(f"score files differ in length: {len(win)} vs {len(one)}")
    recs = load_pool(pool_dir("cuad"), "test")[: len(win)]  # a rehearsal scores a prefix of the pool
    if any(r.question.qid != w["qid"] for r, w in zip(recs, win)):
        raise SystemExit("scores do not align with the test pool")
    tok, _ = load_tokenizer_adapter(*ENCODERS[a.size])
    limit = state_tokens("cuad")  # content tokens one window holds
    offsets = {}
    beyond, inside = [], []
    for i, r in enumerate(recs):
        c = r.meta.get("evidence_start_char")
        if c is None:
            continue
        if r.state_hash not in offsets:
            offsets[r.state_hash] = [o[0] for o in tok(r.state, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
        (beyond if bisect.bisect_right(offsets[r.state_hash], c) - 1 >= limit else inside).append(i)
    pick = lambda rows, idx: [rows[i] for i in idx]
    out = {"n_test": len(recs), "window_content_tokens": limit, "all": paired_compare(win, one),
           "evidence_beyond_window": {"n": len(beyond), **(paired_compare(pick(win, beyond), pick(one, beyond)) if beyond else {})},
           "evidence_inside_window": {"n": len(inside), **(paired_compare(pick(win, inside), pick(one, inside)) if inside else {})}}
    e4_all = out["all"]["accuracy_pp"]["point"] >= 0
    e4_beyond = bool(beyond) and out["evidence_beyond_window"]["accuracy_pp"]["point"] > 0
    out["E4"] = {"all_point_ge_0": e4_all, "beyond_point_gt_0": e4_beyond}
    out["beyond_idx"], out["inside_idx"] = beyond, inside  # test-pool indices, so any arm can be scored on the subsets
    write_json(RESULTS / "mask_price" / f"{a.size}{sfx}-windowed-vs-block" / "beyond_window.json", out)
    for k in ("all", "evidence_beyond_window", "evidence_inside_window"):
        d = out[k]
        if "accuracy_pp" in d:
            print(k, "n", d["n"], "windowed - one window (pp)", round(d["accuracy_pp"]["point"], 2),
                  [round(d["accuracy_pp"]["ci_low"], 2), round(d["accuracy_pp"]["ci_high"], 2)])
