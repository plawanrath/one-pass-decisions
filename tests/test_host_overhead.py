"""The host-overhead reductions on the cold batched path change no output: the cascade kernel's index lists (built with
numpy ranges once per batch), memoised and batched tokenization, the readout from one host copy of the probabilities,
and the cache fill that slices local layers straight to their visible tail. Each is checked against the computation it
replaced (kept here as a reference) to 1e-6 in fp32 on the CPU, or exactly where the values are integers."""
import numpy as np
import pytest
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine, _confidence
from onepass.engine.packer import Group, Pending
from onepass.engine.state_cache import StateEntry, questions_with_cache, trim_local
from onepass.model.backbones import cascade_indices
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import CHUNK_SEP, Layout, TokenizerAdapter, pack_batch
from onepass.schema import Question, Request

QS = [Question("choice", "which domain", ["a", "b", "c"], qid="q1"), Question("noul", "is it safe", ["false", "true"], qid="q2"),
      Question("score", "rate it", ["l1", "l2", "l3", "l4"], qid="q3"), Question("noul", "is it long", ["false", "true"], qid="q4")]
STATES = [" ".join(f"w{i}x{j}" for i in range(n)) for j, n in enumerate((60, 23, 90, 41))]


def reference_cascade_indices(blk, Ss, L, half_window):
    """The list-building loop the cascade kernel used before (one Python list per index set)."""
    local = half_window < (1 << 29)
    gq, gk, gcq, gck = [], [], [0], [0]
    bq, bcq, ia = [], [0], []
    lq, lk, lcq, lck = [], [], [0], [0]
    for b in range(blk.shape[0]):
        S, row, base = int(Ss[b]), blk[b], b * L
        state = list(range(base, base + S))
        ids = row[S:]
        n = int((ids >= 0).sum())
        cuts = (np.flatnonzero(ids[1:n] != ids[:n - 1]) + 1).tolist() if n else []
        blocks = [list(range(base + S + a, base + S + e)) for a, e in zip([0] + cuts, cuts + [n])] if n else []
        gq += state
        for blk_rows in blocks:
            ia += list(range(len(gq), len(gq) + len(blk_rows)))
            gq += blk_rows
            bq += blk_rows
            bcq.append(len(bq))
        gk += state
        gcq.append(len(gq))
        gck.append(len(gk))
        if local:
            tail = state[-half_window:] if half_window > 0 else []
            lq += state
            lk += state
            lcq.append(len(lq))
            lck.append(len(lk))
            for blk_rows in blocks:
                lq += blk_rows
                lk += tail + blk_rows
                lcq.append(len(lq))
                lck.append(len(lk))
    return {"gq": gq, "gk": gk, "bq": bq, "ia": ia, "lq": lq, "lk": lk, "gcq": gcq, "gck": gck, "bcq": bcq, "lcq": lcq, "lck": lck}


@pytest.mark.parametrize("half_window", (0, 3, 8, 64, 1 << 30))
def test_cascade_indices_equal_the_list_reference(tok, half_window):
    rng = np.random.default_rng(half_window % 7)
    lays = []
    for j in range(6):
        k = int(rng.integers(0, 5))  # rows with no questions included
        lays.append(Layout([1] * 0, [], [], [], 0))
        S = int(rng.integers(1, 40))
        ids, blk, pos = [7] * S, [0] * S, list(range(S))
        for q in range(1, k + 1):
            n = int(rng.integers(1, 9))
            ids += [8] * n
            blk += [q] * n
            pos += list(range(S, S + n))
        lays[-1] = Layout(ids, blk, pos, [[len(ids) - 1]] * k, S, [])
    b = pack_batch(lays, tok.pad_id, "block")
    got = cascade_indices(b.block_ids.numpy(), b.state_len.tolist(), b.block_ids.shape[1], half_window)
    ref = reference_cascade_indices(b.block_ids.numpy(), b.state_len.tolist(), b.block_ids.shape[1], half_window)
    for key, val in ref.items():
        assert list(np.asarray(got[key]).tolist()) == val, key


class _BatchTok:
    """A fake fast tokenizer that also encodes batches (as a Hugging Face fast tokenizer does)."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 50281, 50282, 50284, 50283
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        return [100 + sum(map(ord, w)) % 40000 for w in text.split()]

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.encode(t) for t in texts]}


def test_memoised_and_batched_tokenization_give_the_same_ids():
    raw = _BatchTok()
    t = TokenizerAdapter(raw)
    texts = ["is it safe", "a b c", "x " * 400, "is it safe"]
    for x in texts:
        a, b = t.encode(x), t.encode(x)
        assert a == b == raw.encode(x) and a is not b  # a fresh list each call: callers may extend it
    assert t.encode_many(texts) == [raw.encode(x) for x in texts]


def test_real_tokenizer_batch_equals_one_at_a_time():
    try:
        from onepass.cli import ENCODERS
        from onepass.model.encoder import load_tokenizer_adapter

        _, t = load_tokenizer_adapter(*ENCODERS["base"], local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached locally")
    texts = ["This Agreement is governed by the laws of Delaware.", "Receiving Party shall not disclose " * 300,
             "  spaces  and\nnewlines\t", "unicode: café — naïve", ""]
    assert t.encode_many(texts) == [t.encode(x) for x in texts]


@torch.no_grad()
def legacy_decisions(enc, tok, cfg, groups, eng):
    """The cold batched path as it was: per-request layout, one packed stock pass, a softmax per row on the device and
    every probability read on its own."""
    mode = cfg.mask
    lays = [eng._layout(g.state, g.questions) for g in groups]
    logits = enc(pack_batch(lays, tok.pad_id, mode))
    out = {}
    for r, g in enumerate(groups):
        probs = torch.softmax(logits[r], dim=-1)
        for k, (p_, q) in enumerate(zip(g.items, g.questions)):
            p = probs[k, : len(q.options)]
            out[p_.seq] = (int(p.argmax()), [float(x) for x in p], _confidence(p, cfg.confidence))
    return out


def _groups(states, qsets):
    gs = []
    for i, (s, qs) in enumerate(zip(states, qsets)):
        gs.append(Group(Request(s, qs).state_hash, s, [Pending(0.0, i * 100 + k, Request(s, qs), k) for k in range(len(qs))]))
    return gs


def _same(new, ref, tol=1e-6):
    assert set(new) == set(ref)
    for seq, (idx, probs, conf) in ref.items():
        d = new[seq][0]
        assert d.index == idx and abs(d.confidence - conf) <= tol
        assert np.allclose(d.probs, probs, atol=tol, rtol=0)


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
@pytest.mark.parametrize("confidence", ("max_prob", "margin"))
def test_cold_batch_and_hit_decisions_unchanged(tok, family, confidence):
    enc = OnePassEncoder.tiny(seed=2, family=family)
    cfg = EngineConfig(confidence=confidence)
    eng = OnePassEngine(enc, tok, cfg)
    groups = _groups(STATES, [QS, QS[:2], QS[1:], QS[:1]])
    ref = legacy_decisions(enc, tok, cfg, groups, eng)
    cold = eng.decide_groups(groups)  # cold, fills the cache
    assert all(d[0].path.endswith(":cold") for d in cold.values())
    _same(cold, ref)
    hit = eng.decide_groups(groups)  # every state is now cached: the hit path
    assert all(d[0].path.endswith(":hit") for d in hit.values())
    _same(hit, ref, tol=1e-5)  # hit vs packed pass: equal up to float reassociation (as the existing cache tests)


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
def test_fused_trimmed_fill_equals_trim_local_of_the_full_fill(tok, family):
    enc = OnePassEncoder.tiny(seed=3, family=family)
    eng = OnePassEngine(enc, tok, EngineConfig())
    lays = [eng._layout(s, qs) for s, qs in zip(STATES, [QS, QS[:2], QS[1:], QS[:1]])]
    b = pack_batch(lays, tok.pad_id, "block")
    h0, full = enc.hidden_and_states(b)
    h1, fused = enc.hidden_and_states(b, local_tail=True)
    assert torch.equal(h0, h1)
    for (ks, vs, pos), st, S in zip(full, fused, b.state_len.tolist()):
        ref = trim_local(enc, StateEntry("k", S, ks, vs, pos))
        got = StateEntry("k", S, *st) if len(st) == 4 else trim_local(enc, StateEntry("k", S, *st))
        assert all(torch.equal(a, r) for a, r in zip(got.keys + got.values, ref.keys + ref.values))
        assert torch.equal(got.positions, ref.positions)
        assert (got.local_idx is None) == (ref.local_idx is None)
        if ref.local_idx is not None:
            assert torch.equal(got.local_idx.cpu(), ref.local_idx.cpu())
    # the engine's cache entries are the trim of the full fill, so hits answer as before
    eng.decide_groups(_groups(STATES, [QS, QS[:2], QS[1:], QS[:1]]))
    e = eng.cache.get(_groups(STATES[:1], [QS])[0].state_hash)
    ref = trim_local(enc, StateEntry(e.key, e.state_len, *full[0]))
    assert all(torch.equal(a, r) for a, r in zip(e.keys + e.values, ref.keys + ref.values))


def test_cascade_cpu_reference_decisions_match_masked_attention(tok):
    enc = OnePassEncoder.tiny(seed=4)
    groups = _groups(STATES, [QS, QS[:2], QS[1:], QS[:1]])
    base = OnePassEngine(enc, tok, EngineConfig())
    ref = legacy_decisions(enc, tok, base.cfg, groups, base)
    casc = OnePassEngine(enc, tok, EngineConfig(attn_kernel="cascade"))
    _same(casc.decide_groups(groups), ref, tol=1e-5)  # a different (exact) kernel: float reassociation only
    _same(casc.decide_groups(groups), ref, tol=1e-5)  # and from its cache


@pytest.mark.parametrize("mask", ("windowed", "chunked"))
def test_windowed_and_chunked_decisions_unchanged(tok, mask):
    enc = OnePassEncoder.tiny(seed=5)
    if mask == "windowed":
        cfg = EngineConfig(mask="windowed", window_tokens=24, max_state_tokens=96)
        states = STATES
    else:
        cfg = EngineConfig(mask="chunked", max_state_tokens=200)
        states = [CHUNK_SEP.join(" ".join(s.split()[i:i + 9]) for i in range(0, len(s.split()), 9)) for s in STATES]
    eng = OnePassEngine(enc, tok, cfg)
    groups = _groups(states, [QS, QS[:2], QS[1:], QS[:1]])
    ref = legacy_decisions(enc, tok, cfg, groups, eng)
    _same(eng.decide_groups(groups), ref, tol=1e-5)
    _same(eng.decide_groups(groups), ref, tol=1e-5)
