"""The encoder-decoder baseline (scripts/encdec_finetune.py, scripts/encdec_serve.py): the served path, with the contract
encoded once and its cross-attention keys and values cached, gives the same letter logits as the full model run on
each question separately."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from onepass.schema import DecisionRecord, Question  # noqa: E402


@pytest.fixture(scope="module")
def t5_tok():
    from transformers import AutoTokenizer

    from encdec_finetune import ENCDECS

    repo, rev = ENCDECS["t5-base"]
    try:
        return AutoTokenizer.from_pretrained(repo, revision=rev, local_files_only=True)
    except Exception:
        pytest.skip("Flan-T5 tokenizer not cached on this host")


def tiny_t5(seed=0):
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(seed)
    cfg = T5Config(vocab_size=32128, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4,
                   feed_forward_proj="gated-gelu", tie_word_embeddings=False, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1, dropout_rate=0.0)
    return T5ForConditionalGeneration(cfg).eval()


def _questions():
    return [Question("choice", f"Does clause {i} hold?", ["entailed", "contradicted", "not mentioned"][: 2 + i % 2], qid=f"q{i}") for i in range(5)]


def test_letter_ids_are_single_sentencepiece_letters(t5_tok):
    from encdec_finetune import letter_ids

    ids = letter_ids(t5_tok)
    assert t5_tok.convert_ids_to_tokens(ids[:4]) == ["▁A", "▁B", "▁C", "▁D"]
    assert len(set(ids)) == len(ids)


def test_batched_suffixes_and_cached_path_equal_per_question_full_forward(t5_tok):
    from encdec_finetune import cached_letter_logits, cross_kv, encode, encode_context, encode_suffix, letter_ids, letter_logits

    m = tiny_t5(seed=1)
    letters = letter_ids(t5_tok)
    ctx = encode_context(t5_tok, "The recipient shall keep the information confidential. " * 12, 4096)
    sufs = [encode_suffix(t5_tok, q, m.config.decoder_start_token_id) for q in _questions()]
    assert len({len(s) for s in sufs}) > 1  # different lengths: the right padding is exercised
    with torch.no_grad():
        # reference: the stock model, one question at a time (encoder re-run for each)
        ref = torch.stack([m(input_ids=torch.tensor([ctx]), decoder_input_ids=torch.tensor([s])).logits[0, -1, letters] for s in sufs])
        enc = encode(m, ctx, torch.device("cpu"))
        batched = letter_logits(m, enc, sufs, letters)
        cached = cached_letter_logits(m, enc, cross_kv(m, enc), sufs, letters)
    assert torch.allclose(batched, ref, atol=1e-5)
    assert torch.allclose(cached, ref, atol=1e-5)
    # the cached cross-attention keys and values are reused, not recomputed: zeroing the key/value projections after
    # caching leaves the cached path unchanged
    kv = cross_kv(m, enc)
    for blk in m.decoder.block:
        blk.layer[1].EncDecAttention.k.weight.data.zero_()
        blk.layer[1].EncDecAttention.v.weight.data.zero_()
    with torch.no_grad():
        assert torch.allclose(cached_letter_logits(m, enc, kv, sufs, letters), ref, atol=1e-5)


def test_readout_reads_the_option_letters_at_the_last_real_token(t5_tok):
    from encdec_finetune import encode, encode_context, encode_suffix, letter_ids, letter_logits

    m = tiny_t5(seed=2)
    letters = letter_ids(t5_tok)
    sufs = [encode_suffix(t5_tok, q, 0) for q in _questions()]
    with torch.no_grad():
        enc = encode(m, encode_context(t5_tok, "short contract", 4096), torch.device("cpu"))
        lg = letter_logits(m, enc, sufs, letters)
        for k, s in enumerate(sufs):  # column j is letter j's logit at the suffix's own last token (not at the padding)
            full = m(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([s])).logits[0, len(s) - 1]
            assert torch.allclose(lg[k], full[letters], atol=1e-5)
            assert int(lg[k].argmax()) == int(torch.tensor([full[t] for t in letters]).argmax())


def test_scores_round_trip_through_read_scores(t5_tok, tmp_path):
    from encdec_finetune import EncDecTrainConfig, score_encdec, train_encdec

    from onepass.analysis import read_scores
    from onepass.metrics import summary
    from onepass.train.finetune import rows_to_arrays

    recs = []
    for d in range(3):
        state = f"Contract {d}. " + "The parties agree to terms. " * 6
        for i, q in enumerate(_questions()):
            recs.append(DecisionRecord("contractnli", state, q, (d + i) % len(q.options), "dataset", f"doc{d}"))
    m = tiny_t5(seed=3)
    rep = train_encdec(m, t5_tok, recs, EncDecTrainConfig(lr=1e-3, epochs=2, docs_per_step=2, select_points=2, autocast_bf16=False,
                                                          grad_checkpointing=False, log_every=1000), dev_records=recs, log=lambda *a: None)
    assert rep["selected_step"] in [r["step"] for r in rep["selection"]] and rep["steps"] == 3
    rows = score_encdec(m, t5_tok, recs, 4096, tmp_path / "scores_test.jsonl")
    back = read_scores(tmp_path / "scores_test.jsonl")
    assert len(back) == len(recs) and [r["idx"] for r in back] == list(range(len(recs)))
    assert abs(summary(*rows_to_arrays(rows))["accuracy"] - max(r["dev_accuracy"] for r in rep["selection"])) < 1e-9
    assert all(abs(sum(r["probs"]) - 1) < 1e-4 and len(r["probs"]) == len(recs[r["idx"]].question.options) for r in back)


def test_flops_and_cache_accounting():
    from encdec_finetune import cache_bytes_per_token, flops_per_contract, nonemb_params

    m = tiny_t5()
    c = m.config
    enc_p, dec_p = nonemb_params(m)
    assert enc_p > 0 and dec_p > enc_p  # the decoder adds cross-attention to every layer
    assert cache_bytes_per_token(c) == {"encoder_output": 64, "cross_kv": 2 * 4 * 8 * 2 * 2}
    f1 = flops_per_contract(c, enc_p, dec_p, 100, [10])
    f2 = flops_per_contract(c, enc_p, dec_p, 100, [10, 10])
    f0 = flops_per_contract(c, enc_p, dec_p, 100, [])
    assert f2 - f1 == pytest.approx(f1 - f0)  # each question adds the same decoder work; the context is paid once


def test_server_padded_encoder_batch_equals_one_contract_at_a_time(t5_tok, tmp_path):
    from encdec_finetune import cached_letter_logits, cross_kv, encode, letter_ids
    from encdec_serve import EncDecServer

    m = tiny_t5(seed=4)
    m.save_pretrained(tmp_path / "m")
    t5_tok.save_pretrained(tmp_path / "m")
    srv = EncDecServer(str(tmp_path / "m"), "cpu", torch.float32, 4096, enc_batch=2)
    qs = [{"kind": q.kind, "text": q.text, "options": q.options, "qid": q.qid} for q in _questions()]
    reqs = [srv.build("A short contract about confidentiality.", qs), srv.build("A much longer contract. " * 30, qs)]
    assert len(reqs[0][0]) != len(reqs[1][0])
    srv.run_many(reqs)  # both contracts encoded in one padded batch
    letters = letter_ids(t5_tok)
    with torch.no_grad():
        for ctx, sufs, _ in reqs:
            enc_b, kv_b = srv.cache[tuple(ctx)]
            enc = encode(srv.model, ctx, torch.device("cpu"))
            assert torch.allclose(enc_b, enc, atol=1e-5)
            assert torch.allclose(cached_letter_logits(srv.model, enc_b, kv_b, sufs, letters),
                                  cached_letter_logits(srv.model, enc, cross_kv(srv.model, enc), sufs, letters), atol=1e-5)
