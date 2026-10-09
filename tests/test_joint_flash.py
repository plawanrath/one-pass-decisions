"""Joint packing through the variable-length path (reference kernel on CPU) gives the same option logits as the masked
joint forward the engine runs (mask="joint")."""
import torch

from onepass.cli import ENCODERS
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request, pack_batch
from onepass.paths.varlen import joint_logits
from onepass.schema import Question


def test_joint_flash_matches_masked():
    torch.manual_seed(0)
    enc = OnePassEncoder.from_pretrained(*ENCODERS["base"]).eval()
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    state = "The receiving party shall hold the information in confidence. " * 30
    qs = [Question("choice", f"Does clause {k} hold?", ["yes", "no", "unclear"], qid=f"q{k}") for k in range(4)]
    lay = layout_request(tok, state, qs, "full", 4096, 24)
    with torch.no_grad():
        ref = enc(pack_batch([lay], tok.pad_id, "full"))[0]
        got = joint_logits(enc, lay, kernel="reference")
    for k, g in enumerate(got):
        assert torch.allclose(g, ref[k, : len(g)].float(), atol=1e-4), (k, g, ref[k])
