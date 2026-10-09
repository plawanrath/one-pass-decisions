"""Request lists shared by every latency path: the same states and questions for the encoder paths and the decoders."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path


def make_requests(recs, tok, S: int, K: int, n: int, seed: int = 0):
    """States padded/truncated to about S tokens by concatenating pool states; K questions from the pool."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        parts, ntok = [], 0
        while ntok < S:
            r = recs[rng.randrange(len(recs))]
            parts.append(r.state)
            ntok += len(tok.encode(r.state))
        state = "\n".join(parts)
        qs = []
        for k in range(K):
            q = copy.deepcopy(recs[rng.randrange(len(recs))].question)
            q.qid = f"{q.qid}:{k}"
            qs.append(q)
        out.append((state, qs))
    return out


def dump(path: str | Path, cells: dict) -> None:
    """cells: {"S:K": [(state, [Question, ...]), ...]} -> JSON readable without the package's model dependencies."""
    js = {key: [{"state": s, "questions": [{"kind": q.kind, "text": q.text, "options": q.options, "qid": q.qid} for q in qs]}
                for s, qs in reqs] for key, reqs in cells.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(js))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
