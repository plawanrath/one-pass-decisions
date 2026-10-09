"""The mechanism on a second encoder family (EuroBERT) and the windowed state, checked numerically on tiny models.

Every invariance the engine relies on is asserted for both families: the hand-run layer loop equals the stock
forward, state rows do not depend on questions, a question does not depend on the questions packed with it, the cache
hit path equals the cold packed pass, chunk extension equals re-encoding. The windowed mode adds: windows do not see
each other, the two-stage forward equals the dense packed forward (values and gradients), a state that fits one
window is exactly the block layout, and the windowed cache entry reproduces the cold pass.
"""
import pytest
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.state_cache import encode_state, extend_state, manual_forward, questions_with_cache
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import CHUNK_SEP, Layout, TokenizerAdapter, layout_request, pack_batch, windowed_state_ids
from onepass.schema import Question, Request

FAMILIES = ("modernbert", "eurobert")
STATE = " ".join(f"tok{i}" for i in range(40))
LONG = " ".join(f"w{i}" for i in range(57))  # 57 content tokens: windows of 16 -> 14 per window -> 5 windows
QS = [Question("choice", "which domain", ["banking", "travel", "utility"]),
      Question("noul", "the step is safe", ["false", "true"]),
      Question("score", "rate the answer", ["one", "two", "three", "four"])]
W = 16
ATOL = 2e-5


def enc_for(family, seed=0):
    return OnePassEncoder.tiny(seed=seed, family=family).eval()


def win_batch(tok, state, qs, window=W, budget=0):
    return pack_batch([layout_request(tok, state, qs, "windowed", budget, window_tokens=window)], tok.pad_id, "windowed")


def q_only(lay: Layout) -> Layout:
    S = lay.state_len
    return Layout(lay.ids[S:], lay.block_ids[S:], lay.positions[S:], [[p - S for p in m] for m in lay.marker_pos], 0, lay.question_lens)


# ---- tokenizer and layout ----------------------------------------------------------------------------------------
def test_adapter_falls_back_to_bos_eos():
    class NoClsTok:
        cls_token_id = sep_token_id = None
        bos_token_id, eos_token_id, mask_token_id, pad_token_id = 1, 2, 3, 4

        def encode(self, t, add_special_tokens=False):
            return [10] * len(t.split())

    a = TokenizerAdapter(NoClsTok())
    assert (a.cls_id, a.sep_id, a.mask_id, a.pad_id) == (1, 2, 3, 4)


def test_windowed_state_ids_split_and_budget(tok):
    ids, pos, wid = windowed_state_ids(tok, list(range(100, 157)), W)
    assert wid[-1] == 4 and len(ids) == 57 + 2 * 5
    for w in range(5):
        p = [pp for pp, ww in zip(pos, wid) if ww == w]
        i = [ii for ii, ww in zip(ids, wid) if ww == w]
        assert p == list(range(len(p))) and i[0] == tok.cls_id and i[-1] == tok.sep_id and len(i) <= W
    ids2, _, wid2 = windowed_state_ids(tok, list(range(100, 157)), W, max_state_tokens=2 * W)  # two windows' worth
    assert wid2[-1] == 1 and len(ids2) == 2 * W
    ids4, _, _ = windowed_state_ids(tok, list(range(100, 157)), W, 20)  # 20 layout tokens: 2 windows, 16 content tokens
    assert len(ids4) == 20 and ids4[:2] == [tok.cls_id, 100]
    _, _, wid3 = windowed_state_ids(tok, list(range(100, 157)), W, 2 * W, "tail")
    assert ids2 != windowed_state_ids(tok, list(range(100, 157)), W, 2 * W, "tail")[0] and wid3[-1] == 1


def test_windowed_layout_question_positions(tok):
    lay = layout_request(tok, LONG, QS, "windowed", window_tokens=W)
    assert lay.state_len == 57 + 10
    starts = [lay.positions[lay.state_len + sum(lay.question_lens[:k])] for k in range(len(QS))]
    assert starts == [W] * len(QS)  # every question block starts after the longest window


@pytest.mark.parametrize("family", FAMILIES)
def test_one_window_is_the_block_layout(tok, family):
    enc = enc_for(family)
    lb = layout_request(tok, STATE, QS, "block")
    lw = layout_request(tok, STATE, QS, "windowed", window_tokens=64)
    assert (lb.ids, lb.positions, lb.block_ids, lb.marker_pos) == (lw.ids, lw.positions, lw.block_ids, lw.marker_pos)
    with torch.no_grad():
        a = enc(pack_batch([lb], tok.pad_id, "block"))
        b = enc(pack_batch([lw], tok.pad_id, "windowed"))
    f = torch.isfinite(a)
    assert torch.allclose(a[f], b[f], atol=ATOL)


# ---- the second family: stock equivalence and the block-mask invariances -------------------------------------------
def test_eurobert_layer_loop_equals_stock_forward(tok):
    enc = enc_for("eurobert")
    b = pack_batch([layout_request(tok, STATE, QS[:1], "full"), layout_request(tok, "short state", QS[1:2], "full")], tok.pad_id, "full")
    with torch.no_grad():
        ours = enc.hidden(b)
        stock = enc.backbone(input_ids=b.input_ids, attention_mask=b.attention_pad.long()).last_hidden_state
    real = b.attention_pad
    assert torch.allclose(ours[real], stock[real], atol=ATOL)


def test_eurobert_uses_grouped_kv_heads():
    enc = enc_for("eurobert")
    assert enc.backbone.config.num_key_value_heads < enc.backbone.config.num_attention_heads


@pytest.mark.parametrize("family", FAMILIES)
def test_block_invariances(tok, family):
    enc = enc_for(family)
    with torch.no_grad():
        h0 = enc.hidden(pack_batch([layout_request(tok, STATE, [], "block")], tok.pad_id, "block"))
        h3 = enc.hidden(pack_batch([layout_request(tok, STATE, QS, "block")], tok.pad_id, "block"))
        alone = enc(pack_batch([layout_request(tok, STATE, [QS[1]], "block")], tok.pad_id, "block"))[0, 0, :2]
        packed = enc(pack_batch([layout_request(tok, STATE, QS, "block")], tok.pad_id, "block"))[0, 1, :2]
        reord = enc(pack_batch([layout_request(tok, STATE, [QS[2], QS[0], QS[1]], "block")], tok.pad_id, "block"))[0, 2, :2]
    assert torch.allclose(h0[0, :42], h3[0, :42], atol=ATOL)
    assert torch.allclose(alone, packed, atol=ATOL) and torch.allclose(alone, reord, atol=ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_cache_hit_equals_cold(tok, family):
    enc = enc_for(family)
    lay = layout_request(tok, STATE, QS, "block")
    entry = encode_state(enc, pack_batch([layout_request(tok, STATE, [], "block")], tok.pad_id, "block"), "k")
    qb = pack_batch([q_only(lay)], tok.pad_id, "block")
    with torch.no_grad():
        cold = enc(pack_batch([lay], tok.pad_id, "block"))
        hit = enc.logits_from_hidden(questions_with_cache(enc, entry, qb), qb)
    f = torch.isfinite(cold)
    assert torch.allclose(cold[f], hit[f], atol=ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_chunked_extension_equals_reencoding(tok, family):
    enc = enc_for(family)
    chunks = ["the task text", "step one ran", "step two failed badly", "step three"]
    full = pack_batch([layout_request(tok, CHUNK_SEP.join(chunks), [], "chunked")], tok.pad_id, "chunked")
    pre = pack_batch([layout_request(tok, CHUNK_SEP.join(chunks[:2]), [], "chunked")], tok.pad_id, "chunked")
    e_full = encode_state(enc, full, "f")
    e_pre = encode_state(enc, pre, "p")
    n = e_pre.state_len
    lay = layout_request(tok, CHUNK_SEP.join(chunks), [], "chunked")
    new = Layout(lay.ids[n:], [0] * (lay.length - n), list(range(n, lay.length)), [], lay.length - n, [], lay.chunk_ids[n:])
    e_ext = extend_state(enc, e_pre, pack_batch([new], tok.pad_id, "chunked"), "x")
    for a, b in zip(e_full.keys + e_full.values, e_ext.keys + e_ext.values):
        assert torch.allclose(a, b, atol=ATOL)


# ---- the windowed state ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_windowed_two_stage_equals_dense(tok, family):
    enc = enc_for(family)
    b = win_batch(tok, LONG, QS)
    with torch.no_grad():
        two = enc.hidden(b)
        dense = manual_forward(enc, b)
    real = b.attention_pad
    assert torch.allclose(two[real], dense[real], atol=ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_windowed_gradients_equal_dense(tok, family):
    """Training runs the two-stage forward; its gradients must be those of the dense windowed forward."""
    gold = torch.tensor([[1, 0, 2]])
    grads = []
    for dense in (False, True):
        enc = OnePassEncoder.tiny(seed=3, family=family).train()
        b = win_batch(tok, LONG, QS)
        h = manual_forward(enc, b) if dense else enc.hidden(b)
        OnePassEncoder.loss(enc.logits_from_hidden(h, b), gold).backward()
        grads.append({n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) > 0
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-4, rtol=1e-3), n


@pytest.mark.parametrize("family", FAMILIES)
def test_windows_do_not_see_each_other(tok, family):
    enc = enc_for(family)
    other = LONG.split()
    other[-1] = "changed"  # the last window holds only the 57th content token
    with torch.no_grad():
        h1 = enc.hidden(win_batch(tok, LONG, []))
        h2 = enc.hidden(win_batch(tok, " ".join(other), []))
    first4 = 4 * W  # the first four windows are full (16 layout tokens each)
    assert torch.allclose(h1[0, :first4], h2[0, :first4], atol=ATOL)
    assert not torch.allclose(h1[0, first4:67], h2[0, first4:67], atol=ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_windowed_questions_independent_and_read_every_window(tok, family):
    enc = enc_for(family)
    with torch.no_grad():
        alone = enc(win_batch(tok, LONG, [QS[1]]))[0, 0, :2]
        packed = enc(win_batch(tok, LONG, QS))[0, 1, :2]
        other = LONG.split()
        other[3] = "changed"  # a token in the first window
        moved = enc(win_batch(tok, " ".join(other), [QS[1]]))[0, 0, :2]
    assert torch.allclose(alone, packed, atol=ATOL)
    assert not torch.allclose(alone, moved, atol=ATOL)  # the question reads the first window too


@pytest.mark.parametrize("family", FAMILIES)
def test_windowed_cache_hit_equals_cold(tok, family):
    enc = enc_for(family)
    lay = layout_request(tok, LONG, QS, "windowed", window_tokens=W)
    entry = encode_state(enc, win_batch(tok, LONG, []), "k")
    assert entry.state_len == lay.state_len
    qb = pack_batch([q_only(lay)], tok.pad_id, "block")
    with torch.no_grad():
        cold = enc(pack_batch([lay], tok.pad_id, "windowed"))
        hit = enc.logits_from_hidden(questions_with_cache(enc, entry, qb), qb)
    f = torch.isfinite(cold)
    assert torch.allclose(cold[f], hit[f], atol=ATOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_engine_windowed_cold_hit_and_batched(tok, family):
    enc = enc_for(family)
    eng = OnePassEngine(enc, tok, EngineConfig(mask="windowed", window_tokens=W, max_state_tokens=8 * W))
    cold = eng.decide(LONG, QS)
    hit = eng.decide(LONG, QS)
    assert all(d.path.endswith(":cold") for d in cold) and all(d.path.endswith(":hit") for d in hit)
    for a, b in zip(cold, hit):
        assert a.index == b.index and max(abs(x - y) for x, y in zip(a.probs, b.probs)) < 1e-5
    from onepass.engine.packer import Packer

    eng2 = OnePassEngine(enc, tok, EngineConfig(mask="windowed", window_tokens=W, max_state_tokens=8 * W))
    p = Packer()
    p.submit(Request(LONG, QS, "r1"), now=0.0)
    p.submit(Request(STATE, QS[:2], "r2"), now=0.0)
    res, _ = eng2.serve_step(p)
    got = [d for seq in sorted(res) for d in res[seq]]
    ref = OnePassEngine(enc, tok, EngineConfig(mask="windowed", window_tokens=W, max_state_tokens=8 * W, use_cache=False))
    want = ref.decide(LONG, QS) + ref.decide(STATE, QS[:2])
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert max(abs(x - y) for x, y in zip(a.probs, b.probs)) < 1e-5


@pytest.mark.parametrize("family", FAMILIES)
def test_save_load_roundtrip(tmp_path, tok, family):
    enc = enc_for(family)
    enc.save(tmp_path / "m")
    enc2 = OnePassEncoder.load(tmp_path / "m").eval()
    b = pack_batch([layout_request(tok, STATE, QS, "block")], tok.pad_id, "block")
    with torch.no_grad():
        a, c = enc(b), enc2(b)
    f = torch.isfinite(a)
    assert enc2.family == family and torch.allclose(a[f], c[f], atol=1e-6)
