from onepass.gate.certify import Certificate
from onepass.gate.gate import Cascade, Gate, Ledger, Prices
from onepass.schema import Decision, Question

QS = [Question("choice", "a", ["x", "y"], qid="q1"), Question("noul", "b", ["false", "true"], qid="q2")]


def _dec(qid, conf, idx=0):
    return Decision(qid, "choice", idx, [conf, 1 - conf], conf, path="engine")


def test_gate_thresholds_and_roundtrip(tmp_path):
    c = Certificate(0.05, 0.05, 0.8, 100, 60, 2, 0.6, 0.033, 0.01, "g", "p", 5, {"engine": "e"})
    g = Gate.from_certificates({"*": c})
    acc, esc = g.apply([_dec("q1", 0.9), _dec("q2", 0.7)])
    assert [d.qid for d in acc] == ["q1"] and esc[0].escalated
    g.save(tmp_path / "g.json")
    g2 = Gate.load(tmp_path / "g.json")
    assert g2.tau("anything") == 0.8 and g2.certificates["*"]["path"] == {"engine": "e"}
    assert Gate({"q1": 0.5, "*": 0.99}).tau("q1") == 0.5


def test_cascade_escalates_audits_and_accounts():
    calls = []

    def s2(state, q):
        calls.append(q.qid)
        return 1, {"seconds": 0.5, "tokens_in": 100, "tokens_out": 3}

    led = Ledger(Prices(system1_host_usd_per_hour=3.6, system2_host_usd_per_hour=36.0))
    cas = Cascade(Gate({"*": 0.8}), s2, led, audit_rate=1.0)
    res = cas.run("s", QS, [_dec("q1", 0.95, idx=0), _dec("q2", 0.6, idx=0)], s1_seconds_total=0.02)
    assert res[0].source == "system1" and res[0].audited and res[0].audit_disagrees is True
    assert res[1].source == "system2" and res[1].final_index == 1
    assert calls == ["q1", "q2"]
    s = led.summary()
    assert s["n"] == 2 and s["escalate_rate"] == 0.5 and s["audit_rate"] == 0.5
    # usd: s1 0.01 s each at 3.6/h = 1e-5 each; s2 0.5 s at 36/h = 5e-3 twice
    assert abs(s["usd_per_1k"] - 1000 * (2e-5 + 1e-2) / 2) < 1e-9
    a = cas.audit_summary()
    assert a["n_audited"] == 1 and a["disagreement_rate"] == 1.0 and a["confident_disagreement_rate"] == 1.0
