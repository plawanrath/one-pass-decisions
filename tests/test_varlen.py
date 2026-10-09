"""The unpadded variable-length baseline reproduces the padded full-attention forward (both families; reference kernel
on CPU, the PyTorch FlashAttention varlen kernel when CUDA is present)."""
import pytest
import torch

from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import layout_request, pack_batch
from onepass.paths.torch_eager import naive_batch
from onepass.paths.varlen import unpadded_logits, varlen_paths
from onepass.schema import Question

QS = [Question("choice", "which domain", ["banking", "travel", "utility"]),
      Question("noul", "the step is safe", ["false", "true"]),
      Question("score", "rate the answer with a longer question text here", ["one", "two", "three", "four"])]
STATE = " ".join(f"tok{i}" for i in range(40))
KERNELS = ["reference"] + (["torch-varlen"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
@pytest.mark.parametrize("kernel", KERNELS)
def test_unpadded_equals_padded_full(tok, family, kernel):
    dev = "cuda" if kernel == "torch-varlen" else "cpu"
    enc = OnePassEncoder.tiny(seed=1, family=family).eval().to(dev)
    if kernel == "torch-varlen":  # FlashAttention kernels run in half precision; compare against the padded bf16 forward
        enc = enc.to(torch.bfloat16)
    lays = [layout_request(tok, STATE, [q], "full") for q in QS] + [layout_request(tok, "a short state", [QS[0]], "full")]
    with torch.no_grad():
        padded = enc(pack_batch(lays, tok.pad_id, "full").to(dev))
    got = unpadded_logits(enc, lays, kernel)
    tol = 1e-5 if kernel == "reference" else 3e-2
    for r, lg in enumerate(got):
        assert torch.allclose(lg, padded[r, 0, : len(lg)].float(), atol=tol), r


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
def test_varlen_paths_match_naive_batch(tok, family):
    enc = OnePassEncoder.tiny(seed=2, family=family).eval()
    ref = naive_batch(enc, tok, STATE, QS)
    for batched in (False, True):
        got = varlen_paths(enc, tok, STATE, QS, batched)
        assert [d.index for d in got] == [d.index for d in ref]
        assert max(abs(a - b) for x, y in zip(got, ref) for a, b in zip(x.probs, y.probs)) < 1e-5


def test_local_window_is_applied(tok):
    """ModernBERT's local layers must see only |i - j| <= w inside a sequence: with the window removed the result
    changes (so the test above is not passing vacuously)."""
    enc = OnePassEncoder.tiny(seed=3).eval()
    lays = [layout_request(tok, STATE, [QS[0]], "full")]
    a = unpadded_logits(enc, lays, "reference")[0]
    enc.half_window = 1 << 30
    b = unpadded_logits(enc, lays, "reference")[0]
    assert not torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
def test_pretokenized_state_gives_the_same_decisions(tok, family):
    """Baselines given the state's token ids (tokenized once per request) answer exactly as when they tokenize it."""
    from onepass.paths.torch_eager import one_seq

    enc = OnePassEncoder.tiny(seed=5, family=family).eval()
    ids = tok.encode(STATE)
    for fn in (lambda **k: one_seq(enc, tok, STATE, QS, **k), lambda **k: naive_batch(enc, tok, STATE, QS, **k),
               lambda **k: varlen_paths(enc, tok, STATE, QS, True, **k)):
        a, b = fn(), fn(state_ids=ids)
        assert [d.probs for d in a] == [d.probs for d in b]
