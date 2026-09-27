import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("provider", Path(__file__).resolve().parent.parent / "scripts" / "gpu" / "provider.py")
prov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prov)


class FakeHttp:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/pods" and method == "POST":
            if body["gpu"]["id"] == "NVIDIA H100 PCIe":
                raise RuntimeError("POST /pods -> 400: no stock")
            return {"id": "pod1"}
        if path == "/pods/pod1":
            return {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {"22": 10341}}
        if path == "/gputypes":
            return [{"id": "NVIDIA H100 PCIe"}, {"id": "NVIDIA A100"}]
        if path == "/instance-types":
            return {"data": {"gpu_1x_h100_pcie": {"regions_with_capacity_available": [{"name": "us-west-1"}], "instance_type": {"price_cents_per_hour": 329}},
                             "gpu_1x_a10": {"regions_with_capacity_available": []}}}
        if path == "/instance-operations/launch":
            return {"data": {"instance_ids": ["lam1"]}}
        if path == "/instances/lam1":
            return {"data": {"status": "active", "ip": "5.6.7.8"}}
        return {}


def test_runpod_client_shapes():
    h = FakeHttp()
    p = prov.RunPod("tok", http=h)
    iid = p.launch("ssh-ed25519 AAAA test", "n")
    assert iid == "pod1"
    tried = [c[2]["gpu"]["id"] for c in h.calls if c[1] == "/pods" and c[0] == "POST"]
    assert tried == ["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]  # PCIe had no stock, SXM landed
    body = h.calls[-1][2]
    assert body["ports"] == ["22/tcp"] and body["startSsh"] is True and body["env"]["PUBLIC_KEY"].startswith("ssh-ed25519")
    st = p.status("pod1")
    assert (st["state"], st["ip"], st["port"]) == ("RUNNING", "1.2.3.4", 10341)
    p.terminate("pod1")
    assert h.calls[-1][:2] == ("DELETE", "/pods/pod1")


def test_lambda_client_shapes():
    h = FakeHttp()
    p = prov.Lambda("tok", http=h)
    av = p.availability()
    assert av[0]["id"] == "gpu_1x_h100_pcie" and av[0]["regions"] == ["us-west-1"]
    assert p.launch("k", "n") == "lam1"
    assert h.calls[-1][2]["region_name"] == "us-west-1" and h.calls[-1][2]["ssh_key_names"] == ["onepass"]
    assert p.status("lam1")["ip"] == "5.6.7.8"
    p.terminate("lam1")
    assert h.calls[-1] == ("POST", "/instance-operations/terminate", {"instance_ids": ["lam1"]})


def test_ledger_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "LEDGER", tmp_path / "ledger.json")
    prov.ledger_open("runpod", "a", 10, 2.89)
    prov.ledger_close("a")
    d = prov.ledger_load()
    assert d["sessions"][0]["terminated_at"] and prov.ledger_spent(d) < 0.01
    try:
        prov.ledger_open("runpod", "b", 100, 2.89)
        assert False, "cap not enforced"
    except SystemExit as e:
        assert "exceed the cap" in str(e)
