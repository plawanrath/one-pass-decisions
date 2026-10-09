"""The cascade decoder path (onepass.paths.decoder_cascade) equals the per-question prefix-cached decoder: cold and
resident, one contract or several batched, on a small Llama on CPU; and, on CUDA with the pinned SmolLM2-360M in bf16,
the same decisions as the per-question path."""
import pytest
import torch

from onepass.cli import DECODERS
from onepass.decoder import build_doc_sequence, cached_letter_logits, letter_ids
from onepass.paths.decoder_cascade import CascadeDecoder
from onepass.schema import DecisionRecord, Question


@pytest.fixture(scope="module")
def dec_tok():
    from transformers import AutoTokenizer

    repo, rev = DECODERS["smol360"]
    try:
        return AutoTokenizer.from_pretrained(repo, revision=rev, local_files_only=True)
    except Exception:
        pytest.skip("SmolLM2 tokenizer not cached on this host")


def small_llama(tok, seed=0, kv_heads=2):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=kv_heads, max_position_embeddings=2048, attn_implementation="sdpa")
    return LlamaForCausalLM(cfg).eval()


def _recs(d, k):
    state = f"Contract {d}. " + " ".join(f"clause{d}_{i} applies" for i in range(20 + 7 * d))
    qs = [Question("choice", f"Does hypothesis {q} of {d} hold for the parties?" + " really" * q, ["entailed", "contradicted", "not mentioned"], qid=f"h{q}")
          for q in range(k)]
    return [DecisionRecord("t", state, q, 0, "dataset", f"doc{d}") for q in qs]


def _reference(m, tok, recs):
    s = build_doc_sequence(tok, recs, list(range(len(recs))), 4096)
    return s, cached_letter_logits(m, s, letter_ids(tok), torch.device("cpu"))


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_cold_resident_and_batched_equal_per_question(dec_tok, kv_heads):
    m = small_llama(dec_tok, kv_heads=kv_heads)
    cd = CascadeDecoder(m, dec_tok, 4096)
    docs = [_recs(0, 4), _recs(1, 3), _recs(2, 5)]
    refs = [_reference(m, dec_tok, r) for r in docs]
    pre = [cd.prefix_ids(r[0].state) for r in docs]
    sufs = [[cd.suffix_ids(x.question) for x in r] for r in docs]
    for (s, _), p, su in zip(refs, pre, sufs):  # the same token ids the scored decoder sees
        assert s.ids == p + [t for x in su for t in x]
    alone = []
    for (s, ref), p, su in zip(refs, pre, sufs):
        (lg,), (e,) = cd.run([su], prefixes=[p])
        assert torch.allclose(lg, ref, atol=1e-4), (lg - ref).abs().max()
        (hit,), none = cd.run([su], cached=[e])
        assert none is None and torch.allclose(hit, ref, atol=1e-4)
        alone.append(e)
    both, entries = cd.run(sufs, prefixes=pre)  # three contracts in one cold pass
    for lg, (_, ref) in zip(both, refs):
        assert torch.allclose(lg, ref, atol=1e-4)
    hits, _ = cd.run(sufs, cached=entries)  # three resident contracts in one pass
    for lg, (_, ref) in zip(hits, refs):
        assert torch.allclose(lg, ref, atol=1e-4)
    for e, a in zip(entries, alone):
        assert e.length == a.length and all(torch.allclose(x, y, atol=1e-5) for x, y in zip(e.keys, a.keys))


def test_choices_restrict_to_options(dec_tok):
    m = small_llama(dec_tok)
    cd = CascadeDecoder(m, dec_tok, 4096)
    r = _recs(0, 2)
    (lg,), _ = cd.run([[cd.suffix_ids(x.question) for x in r]], prefixes=[cd.prefix_ids(r[0].state)])
    ch = cd.choices(lg, [3, 2])
    assert len(ch[0][1]) == 3 and len(ch[1][1]) == 2 and all(abs(sum(p) - 1) < 1e-5 for _, p in ch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only: FlashAttention variable-length kernels in bf16")
def test_cuda_bf16_decisions_match_per_question_path():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev = DECODERS["smol360"]
    try:
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    except Exception:
        pytest.skip("SmolLM2-360M not reachable")
    cd = CascadeDecoder(m, tok, 4096)
    docs = [_recs(d, 6) for d in range(3)]
    sufs = [[cd.suffix_ids(x.question) for x in r] for r in docs]
    cold, entries = cd.run(sufs, prefixes=[cd.prefix_ids(r[0].state) for r in docs])
    hits, _ = cd.run(sufs, cached=entries)
    agree = total = 0
    for r, lg, hl in zip(docs, cold, hits):
        s = build_doc_sequence(tok, r, list(range(len(r))), 4096)
        ref = cached_letter_logits(m, s, letter_ids(tok), torch.device("cuda"))
        for a, b, c in zip(lg, hl, ref):
            total += 1
            agree += int(int(a[:3].argmax()) == int(c[:3].argmax()) == int(b[:3].argmax()))
        assert (torch.softmax(lg[:, :3], -1) - torch.softmax(ref[:, :3], -1)).abs().max() < 0.05
    assert agree >= total - 1  # bf16: at most one near-tie may flip
