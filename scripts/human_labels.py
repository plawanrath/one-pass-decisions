"""Human labels for the trace-grading questions labelled by System 2 (progress, supervisor): a blind labelling sheet and
the agreement report.

    .venv/bin/python scripts/human_labels.py sheet --n-steps 50 --out <dir>      # labelling page + answer key
    .venv/bin/python scripts/human_labels.py score --labels <exported.json> --key <dir>/key.json
The sheet samples steps from the test pool, stratified by the silver progress level, and asks both questions about each
step, so n steps give 2n labels. The silver labels and the encoder's answers are kept out of the page (key.json only).
The report gives Cohen's kappa (quadratic-weighted for the ordinal progress score) of System 2's silver labels and of
the served encoder against the human labels, with raw agreement beside it.
"""
from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

from onepass.analysis import read_scores
from onepass.cli import RESULTS, SEED, pool_dir
from onepass.tasks.pools import load_pool
from onepass.tasks.tracegrade import LEVELS, SUPERVISOR


def sample_steps(recs, n_steps: int, seed: int = SEED) -> list[str]:
    """Group ids of steps that carry both questions, stratified by the silver progress level (round-robin)."""
    by_step = defaultdict(dict)
    for i, r in enumerate(recs):
        if r.question.qid in ("progress", "supervisor"):
            by_step[(r.group_id, r.meta.get("step"))][r.question.qid] = i
    full = {k: v for k, v in by_step.items() if len(v) == 2}
    strata = defaultdict(list)
    for k, v in full.items():
        strata[recs[v["progress"]].gold].append(k)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    out, levels = [], sorted(strata)
    while len(out) < n_steps and any(strata[l] for l in levels):
        for l in levels:
            if strata[l] and len(out) < n_steps:
                out.append(strata[l].pop())
    return [(k, full[k]) for k in out]


def kappa(a: list[int], b: list[int], k: int, weighted: bool = False) -> float:
    """Cohen's kappa between two raters over k categories; quadratic weights when `weighted`."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1 / n
    pa = [sum(obs[i]) for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2 / (k - 1) ** 2 if weighted else float(i != j)) for j in range(k)] for i in range(k)]
    do = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1.0 - do / de if de else float("nan")


def kappa_ci(a: list[int], b: list[int], k: int, weighted: bool = False, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for kappa, resampling labelled items."""
    rng = random.Random(seed)
    n, vals = len(a), []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        v = kappa([a[i] for i in idx], [b[i] for i in idx], k, weighted)
        if v == v:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else (float("nan"), float("nan"))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trace grading labels</title><style>
body{font:15px/1.45 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px;background:#fafafa;color:#222}
.item{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:18px 0}
pre{white-space:pre-wrap;background:#f4f4f4;padding:10px;border-radius:6px;max-height:420px;overflow:auto;font-size:13px}
.rev{background:#fff7d6}.q{margin:10px 0}.q label{display:inline-block;margin:4px 10px 4px 0}
.bar{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;z-index:2}
button{font-size:15px;padding:6px 14px}.done{color:#2a7}
</style></head><body><div class="bar"><b>Trace grading labels</b> &middot; <span id="count"></span>
&middot; <button onclick="exportJson()">Export labels</button></div>
<p>Each item is one step of a code agent solving a grade-school maths problem. The step under review is highlighted and is
always the last step shown. Answer both questions about <b>that step</b>, from the transcript alone. Your answers are
saved in this browser as you go; press <i>Export labels</i> at any time and send the file back.</p>
<p><b>How well does this step advance the task?</b> poor = wrong direction or broken; weak = little progress or a
mistake that needs undoing; adequate = correct but partial or inefficient; good = correct, clear progress;
excellent = correct and completes or nearly completes the task cleanly.<br><b>What should the supervisor do next?</b>
continue = the step is fine; ask the agent to fix this step = recoverable problem in this step; stop and escalate =
the run should not continue without a human.</p>
%ITEMS%
<script>
const KEY="tracegrade-labels-v1";let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){};count()}
function count(){const n=document.querySelectorAll(".item").length*2;const d=Object.keys(S).length;document.getElementById("count").textContent=d+" of "+n+" answered"}
document.querySelectorAll("input[type=radio]").forEach(el=>{if(S[el.name]===el.value)el.checked=true;el.addEventListener("change",()=>{S[el.name]=el.value;save()})});
function exportJson(){const b=new Blob([JSON.stringify(S,null,1)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="tracegrade_human_labels.json";a.click()}
count();
</script></body></html>"""


def render_item(n: int, step_key: str, state: str) -> str:
    marker = "[STEP UNDER REVIEW]"
    i = state.rfind(marker)
    head, rev = (state[:i], state[i:]) if i >= 0 else (state, "")
    qs = []
    for qid, text, opts in (("progress", "How well does this step advance the task?", LEVELS),
                            ("supervisor", "What should the supervisor do next?", SUPERVISOR)):
        radios = "".join(f'<label><input type="radio" name="{step_key}|{qid}" value="{j}"> {html.escape(o)}</label>' for j, o in enumerate(opts))
        qs.append(f'<div class="q"><b>{html.escape(text)}</b><br>{radios}</div>')
    return (f'<div class="item"><b>Item {n}</b><pre>{html.escape(head)}<span class="rev">{html.escape(rev)}</span></pre>' + "".join(qs) + "</div>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "score"])
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--out", default="labels")
    ap.add_argument("--labels")
    ap.add_argument("--key")
    a = ap.parse_args()
    if a.cmd == "sheet":
        recs = load_pool(pool_dir("tracegrade"), "test")
        enc = read_scores(RESULTS / "path_scores" / "base-tracegrade" / "engine-hit" / "scores_test.jsonl")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        items, key = [], {}
        for n, ((gid, step), idx) in enumerate(sample_steps(recs, a.n_steps), 1):
            sk = f"{gid}:{step}"
            items.append(render_item(n, sk, recs[idx["progress"]].state))
            for qid, i in idx.items():
                key[f"{sk}|{qid}"] = {"silver": recs[i].gold, "encoder": enc[i]["pred"], "k": len(recs[i].question.options)}
        (out / "tracegrade_labels.html").write_text(PAGE.replace("%ITEMS%", "\n".join(items)))
        (out / "key.json").write_text(json.dumps(key, indent=1))
        print(f"wrote {out / 'tracegrade_labels.html'} ({len(items)} steps, {2 * len(items)} labels) and {out / 'key.json'}")
    else:
        lab = json.loads(Path(a.labels).read_text())
        key = json.loads(Path(a.key).read_text())
        rep = {}
        for qid, weighted in (("progress", True), ("supervisor", False)):
            ks = [k for k in key if k.endswith("|" + qid) and k in lab]
            h = [int(lab[k]) for k in ks]
            kk = key[ks[0]]["k"] if ks else 0
            for who in ("silver", "encoder"):
                o = [key[k][who] for k in ks]
                lo, hi = kappa_ci(h, o, kk, weighted)
                rep[f"{qid}/{who}"] = {"n": len(ks), "kappa": kappa(h, o, kk, weighted), "kappa_ci": [lo, hi],
                                       "agreement": sum(x == y for x, y in zip(h, o)) / len(ks) if ks else float("nan"), "weighted": weighted}
        print(json.dumps(rep, indent=1))
        (RESULTS / "human_labels").mkdir(parents=True, exist_ok=True)
        (RESULTS / "human_labels" / "tracegrade_agreement.json").write_text(json.dumps(rep, indent=1) + "\n")
