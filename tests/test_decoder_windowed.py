"""The decoder windowed path (independently prefilled windows with restarted positions, caches concatenated, question
blocks against them) equals one forward under the dense windowed block-causal mask."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_windowed_cache_equals_dense_mask():
    from transformers import LlamaConfig, LlamaForCausalLM

    from decoder_windowed import windowed_letter_logits

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    wins = [torch.randint(0, 64, (n,), generator=g).tolist() for n in (20, 20, 13)]
    sufs = [torch.randint(0, 64, (n,), generator=g).tolist() for n in (5, 7)]
    letters = [3, 4, 5]
    got = windowed_letter_logits(model, None, wins, sufs, letters, torch.device("cpu"))
    ids, pos, spans = [], [], []
    for w in wins:
        ids += w
        pos += list(range(len(w)))
    P, start = len(ids), max(len(w) for w in wins)
    for s in sufs:
        spans.append((len(ids), len(ids) + len(s)))
        ids += s
        pos += list(range(start, start + len(s)))
    N = len(ids)
    allow = torch.zeros(N, N, dtype=torch.bool)
    o = 0
    for w in wins:
        allow[o:o + len(w), o:o + len(w)] = torch.tril(torch.ones(len(w), len(w), dtype=torch.bool))
        o += len(w)
    for s, e in spans:
        allow[s:e, :P] = True
        allow[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
    mask = torch.zeros(1, 1, N, N).masked_fill_(~allow, torch.finfo(torch.float32).min)
    with torch.no_grad():
        h = model.model(input_ids=torch.tensor([ids]), position_ids=torch.tensor([pos]), attention_mask=mask, use_cache=False).last_hidden_state
    want = h[0, [e - 1 for _, e in spans]] @ model.lm_head.weight[torch.tensor(letters)].T
    assert torch.allclose(got, want, atol=1e-5)
