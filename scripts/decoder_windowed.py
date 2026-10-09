"""The decoder counterpart of the served-windowed result: a decoder fine-tuned on one window of CUAD clauses, served under
the windowed structure with no training, and scored on the same test pool and subsets as the encoders.

The structure is the encoder's windowed mask for a decoder (parallel context windows): the rendered context is cut into
windows of the decoder's trained budget, each window is prefilled on its own with positions restarting at zero (causal
within the window, blind to the others), the windows' key-value caches are concatenated, and each question block, at
positions after the longest window, attends to every window and causally to itself. This is computed exactly through
the prefix-cached path (`cached_letter_logits` generalised to several independently prefilled windows). One window is
scored on the same path and host, so the comparison is paired and free of precision differences.

    .venv/bin/python scripts/decoder_windowed.py --model models/_from_h100_flex/decoder-smol360-cuad-lr1e4
Writes results/decoder_windowed/<model>/{scores_test_w1.jsonl, scores_test_w4.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import torch

from onepass.analysis import paired_compare
from onepass.cli import RESULTS, pool_dir, write_json
from onepass.decoder import doc_groups, letter_ids
from onepass.metrics import summary
from onepass.paths.prefill_only import crop_to, render_prefix, render_suffix
from onepass.tasks.pools import load_pool
from onepass.train.finetune import rows_to_arrays


def windows_of(tok, state: str, window: int, n_windows: int, bos_each: bool = False) -> tuple[list[list[int]], list[int]]:
    """The rendered context as up to `n_windows` windows of `window` content tokens (the first also carries BOS, as the
    one-window prefix does), and each content token's character offset within `state` (for the beyond-window subsets)."""
    bos = [tok.bos_token_id] if getattr(tok, "bos_token_id", None) is not None and tok.bos_token_id != tok.eos_token_id else []
    text = render_prefix(state)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    head = text.index(state)
    ids, offs = enc["input_ids"][: window * n_windows], [o[0] - head for o in enc["offset_mapping"]]  # offsets of the whole context
    wins = [ids[i: i + window] for i in range(0, len(ids), window)]
    wins = [bos + w for w in wins] if bos_each else [bos + wins[0]] + wins[1:]
    return wins, offs


@torch.no_grad()
def windowed_letter_logits(model, tok, windows: list[list[int]], suffixes: list[list[int]], letters: list[int], device) -> torch.Tensor:
    from transformers import DynamicCache

    per = []
    for w in windows:  # each window alone, positions from zero
        c = DynamicCache()
        model(input_ids=torch.tensor([w], device=device), position_ids=torch.arange(len(w), device=device)[None], past_key_values=c, use_cache=True)
        per.append(c)
    cache = DynamicCache()
    for li in range(len(per[0].layers)):
        cache.update(torch.cat([c.layers[li].keys for c in per], dim=2), torch.cat([c.layers[li].values for c in per], dim=2), li)
    P, start = cache.get_seq_length(), max(len(w) for w in windows)
    out = []
    for suf in suffixes:
        lg = model(input_ids=torch.tensor([suf], device=device), position_ids=torch.arange(start, start + len(suf), device=device)[None],
                   past_key_values=cache, use_cache=True).logits[0, -1]
        out.append(lg[torch.tensor(letters, device=device)].float())
        crop_to(cache, P)
    return torch.stack(out)


def score(model, tok, recs, window, n_windows, device, out_path, log_every=5, bos_each=False):
    letters = letter_ids(tok)
    rows, beyond = {}, {}
    t0 = time.time()
    groups = doc_groups(recs)
    for gi, g in enumerate(groups):
        wins, offs = windows_of(tok, recs[g[0]].state, window, n_windows, bos_each)
        sufs = [tok.encode(render_suffix(recs[i].question), add_special_tokens=False) for i in g]
        lg = windowed_letter_logits(model, tok, wins, sufs, letters, device)
        for k, ri in enumerate(g):
            r = recs[ri]
            n = len(r.question.options)
            p = torch.softmax(lg[k, :n], dim=-1)
            rows[ri] = {"idx": ri, "qid": r.question.qid, "kind": r.question.kind, "gold": r.gold, "pred": int(p.argmax()),
                        "conf": float(p.max()), "probs": [round(float(x), 6) for x in p], "group_id": r.group_id, "label_source": r.label_source}
            c = r.meta.get("evidence_start_char")
            if c is not None:  # token index of the evidence start in the full rendered context
                beyond[ri] = bisect.bisect_right(offs, c) - 1
        if (gi + 1) % log_every == 0:
            print(f"[dec-windowed] {n_windows} window(s): {gi + 1}/{len(groups)} documents, {time.time() - t0:.0f}s", flush=True)
    out = [rows[i] for i in range(len(recs))]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in out))
    return out, beyond


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=8000, help="tokens per window: the decoder's trained context budget")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--max-eval", type=int, default=None, help="rehearsals: first N test records")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--bos-each", action="store_true", help="every window starts with the begin token (a per-window attention sink), not only the first")
    ap.add_argument("--from-scores", action="store_true", help="reuse the saved score files and recompute only the subsets and comparisons")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").to(a.device).eval()
    recs = load_pool(pool_dir("cuad"), "test")[: a.max_eval or None]
    name = Path(a.model).name
    res = RESULTS / "decoder_windowed" / (name + ("-rh" if a.max_eval else ""))
    if a.from_scores:
        load_rows = lambda f: [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        one, four = load_rows(res / "scores_test_w1.jsonl"), load_rows(res / f"scores_test_w{a.windows}{'-bos' if a.bos_each else ''}.jsonl")
        tok_idx = {}
        for g in doc_groups(recs):
            _, offs = windows_of(tok, recs[g[0]].state, a.window, a.windows)
            for ri in g:
                c = recs[ri].meta.get("evidence_start_char")
                if c is not None:
                    tok_idx[ri] = bisect.bisect_right(offs, c) - 1
    elif a.bos_each and (res / "scores_test_w1.jsonl").exists():  # one window is the same with or without the flag
        one = [json.loads(x) for x in (res / "scores_test_w1.jsonl").read_text().splitlines() if x.strip()]
        tok_idx = {}
        for g in doc_groups(recs):
            _, offs = windows_of(tok, recs[g[0]].state, a.window, a.windows)
            for ri in g:
                c = recs[ri].meta.get("evidence_start_char")
                if c is not None:
                    tok_idx[ri] = bisect.bisect_right(offs, c) - 1
        four, _ = score(model, tok, recs, a.window, a.windows, a.device, res / f"scores_test_w{a.windows}-bos.jsonl", bos_each=True)
    else:
        one, tok_idx = score(model, tok, recs, a.window, 1, a.device, res / "scores_test_w1.jsonl")
        four, _ = score(model, tok, recs, a.window, a.windows, a.device, res / f"scores_test_w{a.windows}{'-bos' if a.bos_each else ''}.jsonl", bos_each=a.bos_each)
    beyond = [i for i, t in tok_idx.items() if t >= a.window]
    inside = [i for i, t in tok_idx.items() if t < a.window]
    pick = lambda rows, idx: [rows[i] for i in sorted(idx)]
    out = {"model": name, "window_tokens": a.window, "windows": a.windows, "n_test": len(recs),
           "one_window": summary(*rows_to_arrays(one)), "windowed": summary(*rows_to_arrays(four)), "against_one_window": {}}
    for sub, idx in (("all", None), ("evidence_beyond_window", beyond), ("evidence_inside_window", inside)):
        x, y = (four, one) if idx is None else (pick(four, idx), pick(one, idx))
        if not x:
            continue
        c = paired_compare(x, y, cluster=True)
        out["against_one_window"][sub] = {"n": len(x), "accuracy_pp": c["accuracy_pp"], "aurc": c["aurc"]}
        print(f"[dec-windowed] {name} four windows minus one on {sub} (n={len(x)}): {c['accuracy_pp']['point']:+.1f} "
              f"[{c['accuracy_pp']['ci_low']:+.1f}, {c['accuracy_pp']['ci_high']:+.1f}] pp", flush=True)
    out["beyond_idx"], out["inside_idx"] = sorted(beyond), sorted(inside)
    out["bos_each_window"] = a.bos_each
    write_json(res / ("summary-bos.json" if a.bos_each else "summary.json"), out)
