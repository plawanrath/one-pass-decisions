"""A decoder served with prefix caching on the request lists the encoder paths are timed on (vLLM, or HF as fallback).

    .venv-vllm/bin/python scripts/decoder_serve.py --backend vllm --model models/decoder-smol1.7-contractnli --task contractnli \
        --cells 2048:17 4096:17 --n 50 --warmup 10 --throughput 1 4 16 64
    .venv-vllm/bin/python scripts/decoder_serve.py --backend vllm --mistral <dir> --task cuad --cells 8192:41 32768:41
Requests come from data/cache/requests/<task>.json (scripts/dump_requests.py), so every path sees the same contracts
and questions. A fine-tuned SmolLM2 is prompted exactly as it was trained (onepass.decoder: context prefix, then one
question block whose last token is read over the option letters); the System 2 model with its chat prompt, context
first. Per request, three timings:
  cold-batch   prefix cache emptied, all K prompts submitted together;
  cold-first   prefix cache emptied, the first prompt alone (it fills the cache) then the other K-1 together;
  warm         the contract's prefix already cached (it was just served), all K prompts together.
Throughput: N distinct contracts with K questions each submitted at once from an empty cache, contracts per second
(median of 3). Agreement: on test documents, the served answers against the packed scoring's answers of the same
fine-tune (results/decoder_finetune/<name>/scores_test.jsonl). Writes results/latency/<host>-cuda-bf16-decoder-<name>-<task>/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from onepass.bench.timing import TimingCell, host_facts
from onepass.cli import REPO, RESULTS, host_tag, write_json

SYSTEM = ("You are a careful grader. Answer typed questions about the given context. "
          "Reply with the number of the single best option and nothing else. Do not explain.")


class Prompter:
    """Token prompts for one request: a shared prefix and one suffix per question, plus the answer tokens allowed."""

    def __init__(self, encode, bos: list[int], letters: list[int], mistral: bool, max_prefix: int):
        self.encode, self.bos, self.letters, self.mistral, self.max_prefix = encode, bos, letters, mistral, max_prefix

    def prefix(self, state: str) -> list[int]:
        if self.mistral:
            head = self.encode(f"[SYSTEM_PROMPT]{SYSTEM}[/SYSTEM_PROMPT][INST]Context:\n")
            return self.bos + head + self.encode(state)[: self.max_prefix]
        from onepass.paths.prefill_only import render_prefix

        return self.bos + self.encode(render_prefix(state))[: self.max_prefix]

    def suffix(self, q: dict) -> list[int]:
        if self.mistral:
            opts = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(q["options"]))
            return self.encode(f"\n\nQuestion: {q['text']}\nOptions:\n{opts}\n\nAnswer with the option number only.[/INST]")
        from onepass.paths.prefill_only import render_suffix
        from onepass.schema import Question

        return self.encode(render_suffix(Question(q["kind"], q["text"], q["options"], qid=q.get("qid", ""))))

    def request(self, state: str, questions: list[dict]) -> tuple[list[list[int]], list[list[int]], int]:
        pre = self.prefix(state)
        return [pre + self.suffix(q) for q in questions], [self.letters[: len(q["options"])] for q in questions], len(pre)


class VLLMBackend:
    def __init__(self, path: str, max_model_len: int, gpu_mem: float, mistral: bool):
        from vllm import LLM

        kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if mistral else {}
        self.llm = LLM(model=path, dtype="bfloat16", enable_prefix_caching=True, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem, seed=0, **kw)
        self.tok = self.llm.get_tokenizer()
        self.name = "vllm"

    def encode(self, text: str) -> list[int]:
        try:
            return list(self.tok.encode(text, add_special_tokens=False))
        except TypeError:  # the Mistral tokenizer wrapper takes no keyword
            return list(self.tok.encode(text, bos=False, eos=False))

    def run(self, prompts: list[list[int]], allowed: list[list[int]] | None, prefix_len: int = 0) -> list[int]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sps = [SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=a) if a else SamplingParams(max_tokens=1, temperature=0.0)
               for a in (allowed or [None] * len(prompts))]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False)
        return [o.outputs[0].token_ids[0] for o in outs]

    def run_many(self, reqs: list[tuple[list[list[int]], list[list[int]], int]]) -> list[int]:
        """Several contracts' prompts in one submission (the scheduler batches them)."""
        return self.run([p for ps, _, _ in reqs for p in ps], [x for _, als, _ in reqs for x in als])

    def reset(self) -> None:
        self.llm.reset_prefix_cache()


class HFBackend:
    """Fallback when vLLM does not install: HF with an explicit prefix cache per contract (one prefill of the prefix,
    then the K suffixes as one batch against the expanded cache)."""

    def __init__(self, path: str, device: str = "cuda"):
        import torch
        if device == "cpu":
            torch.set_num_threads(4)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.dev = torch.device(device)
        self.cache: dict[tuple, object] = {}
        self.name = "hf"

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def run(self, prompts, allowed, prefix_len: int = 0):
        import copy

        torch = self.torch
        from transformers import DynamicCache

        P = prefix_len
        key = tuple(prompts[0][:P])
        with torch.no_grad():
            if key not in self.cache:
                c = DynamicCache()
                self.model(input_ids=torch.tensor([list(key)], device=self.dev), past_key_values=c, use_cache=True)
                self.cache[key] = c
            out = []
            for p, a in zip(prompts, allowed or [None] * len(prompts)):
                c = copy.deepcopy(self.cache[key])
                suf = p[P:]
                lg = self.model(input_ids=torch.tensor([suf], device=self.dev), position_ids=torch.arange(P, P + len(suf), device=self.dev)[None],
                                past_key_values=c, use_cache=True).logits[0, -1]
                out.append(int(a[int(lg[torch.tensor(a, device=self.dev)].argmax())]) if a else int(lg.argmax()))
        return out

    def run_many(self, reqs):
        return [t for ps, als, P in reqs for t in self.run(ps, als, P)]

    def reset(self):
        self.cache.clear()


class HFTightBackend(HFBackend):
    """The tightest prefix-cached decoder path we can write with HF: the prefix prefilled once per contract and kept, then
    all K suffixes as one right-padded batch against it (the cached keys and values expanded to the batch without a
    copy), each answer read at its suffix's last real token. Equal to the per-suffix path (tests)."""

    def __init__(self, path: str, device: str = "cuda"):
        super().__init__(path, device)
        self.name = "hf-tight"

    def run(self, prompts, allowed, prefix_len: int = 0):
        return tight_letters(self.model, prompts, allowed, prefix_len, self.cache, self.dev)


def tight_letters(model, prompts, allowed, P: int, cache_store: dict, dev) -> list[int]:
    """One batched forward of every suffix against the shared cached prefix; returns the chosen answer token per prompt."""
    import torch

    rows = tight_logits(model, prompts, P, cache_store, dev)
    out = []
    for r, a in zip(rows, allowed or [None] * len(prompts)):
        out.append(int(a[int(r[torch.tensor(a, device=dev)].argmax())]) if a else int(r.argmax()))
    return out


def tight_logits(model, prompts, P: int, cache_store: dict, dev):
    """Next-token logits (B, V) at each prompt's last token: the prefix prefilled once and cached, the suffixes batched."""
    import torch
    from transformers import DynamicCache

    key = tuple(prompts[0][:P])
    with torch.no_grad():
        if key not in cache_store:
            c = DynamicCache()
            model(input_ids=torch.tensor([list(key)], device=dev), past_key_values=c, use_cache=True)
            cache_store[key] = [(k, v) for k, v in zip(*_cache_kv(c))]
        kv = cache_store[key]
        sufs = [p[P:] for p in prompts]
        B, L = len(sufs), max(len(x) for x in sufs)
        pad = getattr(model.config, "pad_token_id", None) or 0
        ids = torch.full((B, L), pad, dtype=torch.long, device=dev)
        am = torch.zeros((B, P + L), dtype=torch.long, device=dev)
        am[:, :P] = 1
        for i, x in enumerate(sufs):
            ids[i, :len(x)] = torch.tensor(x, device=dev)
            am[i, P:P + len(x)] = 1
        c = DynamicCache()
        for li, (k, v) in enumerate(kv):
            c.update(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1), li)
        pos = torch.arange(P, P + L, device=dev)[None].expand(B, -1)
        lg = model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=c, use_cache=True).logits
        last = torch.tensor([len(x) - 1 for x in sufs], device=dev)
        return lg[torch.arange(B, device=dev), last]


def _cache_kv(c):
    """Per-layer (keys, values) of a DynamicCache across transformers versions."""
    if hasattr(c, "layers"):
        return [l.keys for l in c.layers], [l.values for l in c.layers]
    return list(c.key_cache), list(c.value_cache)


def letters_for(backend, mistral: bool) -> list[int]:
    if mistral:
        return [backend.encode(str(i + 1))[-1] for i in range(9)]
    from onepass.paths.prefill_only import LETTERS

    return [backend.encode(" " + L)[-1] for L in LETTERS]


def bos_for(backend, mistral: bool) -> list[int]:
    tok = backend.tok
    b = getattr(tok, "bos_token_id", None)
    if mistral:
        return [b if b is not None else 1]
    e = getattr(tok, "eos_token_id", None)
    return [b] if b is not None and b != e else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "hf-tight"])
    ap.add_argument("--model", default=None, help="a fine-tuned decoder directory (scripts/decoder_finetune.py)")
    ap.add_argument("--mistral", default=None, help="a Mistral Small 3.2 directory in its native format (zero-shot)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-prefix", type=int, default=None, help="prefix token cap; default min(max(S, 4096), 8190), within the model's positions")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--throughput", type=int, nargs="*", default=[])
    ap.add_argument("--agreement-docs", type=int, default=20)
    ap.add_argument("--device", default="cuda", help="HF backend only (rehearsals on a workstation use cpu)")
    ap.add_argument("--requests", default=None, help="default: data/cache/requests/<task>.json")
    a = ap.parse_args()
    mistral = a.mistral is not None
    path = a.mistral or a.model
    name = "mistral-small-3.2" if mistral else Path(path).name
    backend = (VLLMBackend(path, a.max_model_len, a.gpu_mem, mistral) if a.backend == "vllm"
               else HFTightBackend(path, a.device) if a.backend == "hf-tight" else HFBackend(path, a.device))
    letters, bos = letters_for(backend, mistral), bos_for(backend, mistral)
    reqs_all = json.loads(Path(a.requests or REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())
    out_dir = RESULTS / "latency" / (f"{host_tag()}-cuda-bf16-decoder-{name}-{a.task}" + ("" if a.backend == "vllm" else f"-{a.backend}"))
    meta = {"backend": a.backend, "model": name, "task": a.task, "max_model_len": a.max_model_len, "cells": {}}
    try:
        import vllm

        meta["vllm"] = vllm.__version__
    except ImportError:
        pass
    for cell in a.cells:
        S, K = (int(x) for x in cell.split(":"))
        reqs = reqs_all[cell]
        cap = a.max_prefix or min(min(max(S, 4096), 8190), a.max_model_len - 192)
        pr = Prompter(backend.encode, bos, letters, mistral, cap)
        built = [pr.request(r["state"], r["questions"]) for r in reqs]
        n_here = min(a.n, len(built) - a.warmup)
        meta["cells"][cell] = {"prefix_cap": cap, "prefix_tokens_median": statistics.median(len(pr.prefix(r["state"])) for r in reqs)}

        def cold_batch(i):
            backend.reset()
            t0 = time.perf_counter()
            backend.run(*built[i])
            return (time.perf_counter() - t0) * 1000

        def cold_first(i):
            backend.reset()
            p, al, P = built[i]
            t0 = time.perf_counter()
            backend.run(p[:1], al[:1], P)
            backend.run(p[1:], al[1:], P)
            return (time.perf_counter() - t0) * 1000

        def warm(i):
            backend.reset()
            p, al, P = built[i]
            backend.run(p[:1], al[:1], P)  # untimed: the contract's prefix is now cached
            t0 = time.perf_counter()
            backend.run(p, al, P)
            return (time.perf_counter() - t0) * 1000

        for label, fn in (("decoder-cold-batch", cold_batch), ("decoder-cold-first", cold_first), ("decoder-warm", warm)):
            f = out_dir / f"cell_S{S}_K{K}_{label}.json"
            if f.exists():
                continue
            for i in range(a.warmup):
                fn(i)
            lat = [fn(a.warmup + i) for i in range(n_here)]
            arr = np.asarray(lat)
            c = TimingCell(label, host_tag() + "-cuda-bf16", 0, S, K, len(reqs[0]["questions"][0]["options"]), len(lat), a.warmup, lat,
                           float(np.percentile(arr, 50)), float(np.percentile(arr, 90)), float(np.percentile(arr, 99)), float(arr.mean()),
                           float(np.percentile(arr, 50)) / K, 1000.0 * K / float(arr.mean()), host_facts(), {"backend": a.backend, "model": name})
            c.save(f)
            print(f"S={S} K={K} {label}: p50 {c.p50_ms:.1f} ms", flush=True)
        if a.throughput:
            tp = {}
            for N in a.throughput:
                if N > len(built):
                    continue
                runs = {"batch": [], "first": []}
                for rep in range(3):
                    sel = [built[(rep * N + j) % len(built)] for j in range(N)]
                    backend.reset()
                    t0 = time.perf_counter()
                    backend.run_many([(ps, als, P) for ps, als, P in sel])
                    runs["batch"].append(N / (time.perf_counter() - t0))
                    backend.reset()
                    t0 = time.perf_counter()
                    backend.run_many([(ps[:1], als[:1], P) for ps, als, P in sel])
                    backend.run_many([(ps[1:], als[1:], P) for ps, als, P in sel])
                    runs["first"].append(N / (time.perf_counter() - t0))
                tp[str(N)] = {k: statistics.median(v) for k, v in runs.items()}
                print(f"S={S} K={K} throughput N={N}: {tp[str(N)]['batch']:.2f} / {tp[str(N)]['first']:.2f} contracts/s", flush=True)
            write_json(out_dir / f"throughput_S{S}_K{K}.json", tp)
    docs = reqs_all.get("_test_docs", [])[: a.agreement_docs]
    ref = RESULTS / "decoder_finetune" / name / "scores_test.jsonl"
    if docs and not mistral and ref.exists():
        ref_pred = {json.loads(line)["idx"]: json.loads(line)["pred"] for line in ref.read_text().splitlines() if line.strip()}
        cap = min(4096 if a.task == "contractnli" else 8000, a.max_model_len - 192)
        pr = Prompter(backend.encode, bos, letters, False, cap)
        same = total = 0
        for d in docs:
            prompts, al, P = pr.request(d["state"], d["questions"])
            toks = backend.run(prompts, al, P)
            for i, t, allowed in zip(d["idx"], toks, al):
                total += 1
                same += int(allowed.index(t) == ref_pred[i])
        meta["agreement_with_packed_scoring"] = {"same": same, "total": total, "rate": same / max(1, total)}
        print(f"[decoder] served vs packed answers agree on {same}/{total}", flush=True)
    write_json(out_dir / "meta.json", meta)
