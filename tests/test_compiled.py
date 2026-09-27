import numpy as np
import pytest
import torch

from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import layout_request, pack_batch
from onepass.paths.compiled import emit as em
from onepass.paths.compiled import reference as ref
from onepass.schema import Question

STATE = " ".join(f"tok{i}" for i in range(30))
QS = [Question("choice", "which domain", ["banking", "travel", "utility"]), Question("noul", "safe", ["false", "true"])]


def _layout(tok):
    return layout_request(tok, STATE, QS, "block")


def test_reference_tracks_torch_and_block_invariance(tok):
    enc = OnePassEncoder.tiny(seed=3, intermediate_size=64).eval()
    lay = _layout(tok)
    markers = [m for ms in lay.marker_pos for m in ms]
    with torch.no_grad():
        want = enc(pack_batch([lay], tok.pad_id, "block"))[0]
    want = torch.cat([want[0, :3], want[1, :2]]).numpy()
    # float mode: the reference's layout, masks, rotary and head equal the PyTorch path
    p32 = ref.pack(enc, bits=32)
    got32 = ref.forward(p32, lay.ids, lay.positions, lay.block_ids, markers)
    assert np.allclose(got32, want, atol=2e-6)
    # 8-bit mode runs and keeps block invariance: the second question alone equals its packed scores
    p = ref.pack(enc, bits=8)
    got = ref.forward(p, lay.ids, lay.positions, lay.block_ids, markers)
    assert np.isfinite(got).all()
    lay1 = layout_request(tok, STATE, QS[1:], "block")
    alone = ref.forward(p, lay1.ids, lay1.positions, lay1.block_ids, lay1.marker_pos[0])
    assert np.array_equal(alone, got[3:])


@pytest.mark.slow
def test_emitted_kernel_is_bit_exact(tok, tmp_path):
    enc = OnePassEncoder.tiny(seed=3, intermediate_size=64).eval()
    p = ref.pack(enc, bits=8)
    em.emit(p, tmp_path / "k")
    exe = em.compile_cli(tmp_path / "k")
    lay = _layout(tok)
    markers = [m for ms in lay.marker_pos for m in ms]
    want = ref.forward(p, lay.ids, lay.positions, lay.block_ids, markers)
    for threads in (1, 4):
        out = em.run_cli(exe, [em.format_line(lay.ids, lay.positions, lay.block_ids, markers)], threads=threads)
        got = em.parse_line(out[0])
        assert np.array_equal(got, want), (threads, got, want)
    cp = em.CompiledPath(exe, threads=2)
    assert np.array_equal(cp.scores(lay.ids, lay.positions, lay.block_ids, markers), want)
    cp.close()
    # 4-bit build compiles and runs
    p4 = ref.pack(enc, bits=4)
    em.emit(p4, tmp_path / "k4")
    exe4 = em.compile_cli(tmp_path / "k4")
    want4 = ref.forward(p4, lay.ids, lay.positions, lay.block_ids, markers)
    got4 = em.parse_line(em.run_cli(exe4, [em.format_line(lay.ids, lay.positions, lay.block_ids, markers)])[0])
    assert np.array_equal(got4, want4)
