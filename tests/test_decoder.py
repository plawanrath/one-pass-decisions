"""The decoder baseline: packed block-causal scoring equals prefix-cached serving; training and scoring run."""
import pytest
import torch

from onepass.cli import DECODERS
from onepass.decoder import (DecoderTrainConfig, block_causal_allowed, build_doc_sequence, cached_letter_logits, doc_groups,
                             kv_bytes_per_token, letter_ids, packed_letter_logits, score_decoder, train_decoder)
from onepass.schema import DecisionRecord, Question


@pytest.fixture(scope="module")
def dec_tok():
    from transformers import AutoTokenizer

    repo, rev = DECODERS["smol360"]
    try:
        return AutoTokenizer.from_pretrained(repo, revision=rev, local_files_only=True)
    except Exception:
        pytest.skip("SmolLM2 tokenizer not cached on this host")


def tiny_llama(tok, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024, attn_implementation="sdpa")
    return LlamaForCausalLM(cfg).eval()


def _recs(n_docs=3, k=4):
    out = []
    for d in range(n_docs):
        state = f"Contract {d}. " + " ".join(f"clause{d}_{i}" for i in range(30))
        for q in range(k):
            out.append(DecisionRecord("t", state, Question("choice", f"Does hypothesis {q} hold?", ["entailed", "contradicted", "not mentioned"], qid=f"h{q}"),
                                      (d + q) % 3, "dataset", f"doc{d}"))
    return out


def test_mask_is_block_causal(dec_tok):
    recs = _recs(1, 2)
    s = build_doc_sequence(dec_tok, recs, [0, 1], 64)
    a = block_causal_allowed(s)
    P, (s0, e0), (s1, e1) = s.prefix_len, *s.block_spans
    assert a[:P, :P].equal(torch.tril(torch.ones(P, P, dtype=torch.bool)))
    assert a[s1:e1, s0:e0].sum() == 0 and a[s0:e0, s1:e1].sum() == 0  # blocks never see each other
    assert a[s0:e0, :P].all() and a[s1:e1, :P].all()
    assert s.pos[s0] == P and s.pos[s1] == P  # every block's positions continue from the prefix


def test_packed_equals_prefix_cached_and_order_invariant(dec_tok):
    m = tiny_llama(dec_tok)
    recs = _recs(1, 4)
    L = letter_ids(dec_tok)
    s = build_doc_sequence(dec_tok, recs, [0, 1, 2, 3], 256)
    (packed,) = packed_letter_logits(m, [s], L, torch.device("cpu"))
    cached = cached_letter_logits(m, s, L, torch.device("cpu"))
    assert torch.allclose(packed, cached, atol=1e-4), (packed - cached).abs().max()
    s_rev = build_doc_sequence(dec_tok, recs, [3, 2, 1, 0], 256)
    (rev,) = packed_letter_logits(m, [s_rev], L, torch.device("cpu"))
    assert torch.allclose(rev.flip(0), packed, atol=1e-4)
    # batching two documents of different lengths with padding changes nothing
    recs2 = _recs(2, 4)
    s2 = build_doc_sequence(dec_tok, recs2, [4, 5, 6], 256)
    both = packed_letter_logits(m, [s, s2], L, torch.device("cpu"))
    assert torch.allclose(both[0], packed, atol=1e-4)
    (alone2,) = packed_letter_logits(m, [s2], L, torch.device("cpu"))
    assert torch.allclose(both[1], alone2, atol=1e-4)


def test_training_lowers_loss_and_scoring_covers_every_record(dec_tok, tmp_path):
    m = tiny_llama(dec_tok, seed=1)
    recs = _recs(3, 4)
    assert [len(g) for g in doc_groups(recs)] == [4, 4, 4]
    cfg = DecoderTrainConfig(lr=3e-3, epochs=8, docs_per_step=2, max_state_tokens=128, grad_checkpointing=False, log_every=1000)
    rep = train_decoder(m, dec_tok, recs, cfg, log=lambda *a: None)
    assert rep["steps"] == 12 and rep["loss_curve"][-1] < rep["loss_curve"][0]
    rows = score_decoder(m, dec_tok, recs, 128, tmp_path / "s.jsonl")
    assert len(rows) == len(recs) and all(abs(sum(r["probs"]) - 1) < 1e-4 and len(r["probs"]) == 3 for r in rows)
    assert (tmp_path / "s.jsonl").read_text().count("\n") == len(recs)


def test_kv_bytes_per_token():
    from transformers import LlamaConfig

    c = LlamaConfig(hidden_size=2048, num_hidden_layers=24, num_attention_heads=32, num_key_value_heads=32)
    assert kv_bytes_per_token(c) == 24 * 2 * 2048 * 2
    c = LlamaConfig(hidden_size=5120, num_hidden_layers=40, num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    assert kv_bytes_per_token(c) == 40 * 2 * 8 * 128 * 2


def test_tight_batched_suffixes_equal_per_suffix(dec_tok):
    """The tight decoder path (prefix cached once, every suffix in one right-padded batch) gives each suffix the same
    next-token logits as running it alone against a fresh copy of the prefix cache."""
    import copy
    import sys
    from pathlib import Path
    from transformers import DynamicCache
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from decoder_serve import tight_logits
    model = tiny_llama(dec_tok, seed=3)
    prefix = dec_tok.encode("Context: " + " ".join(f"w{i}" for i in range(40)), add_special_tokens=False)
    sufs = [dec_tok.encode(t, add_special_tokens=False) for t in (" Q1? A", " A longer second question here? A", " q3")]
    prompts = [prefix + s for s in sufs]
    P = len(prefix)
    got = tight_logits(model, prompts, P, {}, torch.device("cpu"))
    with torch.no_grad():
        base = DynamicCache()
        model(input_ids=torch.tensor([prefix]), past_key_values=base, use_cache=True)
        for i, s in enumerate(sufs):
            c = copy.deepcopy(base)
            ref = model(input_ids=torch.tensor([s]), position_ids=torch.arange(P, P + len(s))[None], past_key_values=c, use_cache=True).logits[0, -1]
            assert torch.allclose(got[i], ref, atol=1e-4), i
