from onepass.gate.risk_control import ThresholdRecord
from onepass.gate.gate import Cascade, Gate, Ledger, Prices
from onepass.schema import Decision, Question

QS = [Question("choice", "a", ["x", "y"], qid="q1"), Question("noul", "b", ["false", "true"], qid="q2")]


def _dec(qid, conf, idx=0):
    return Decision(qid, "choice", idx, [conf, 1 - conf], conf, path="engine")


def test_gate_thresholds_and_roundtrip(tmp_path):
    c = ThresholdRecord(0.05, 0.05, 0.8, 100, 60, 2, 0.6, 0.033, 0.01, "g", "p", 5, {"engine": "e"})
    g = Gate.from_records({"*": c})
    acc, esc = g.apply([_dec("q1", 0.9), _dec("q2", 0.7)])
    assert [d.qid for d in acc] == ["q1"] and esc[0].escalated
    g.save(tmp_path / "g.json")
    g2 = Gate.load(tmp_path / "g.json")
    assert g2.tau("anything") == 0.8 and g2.records["*"]["path"] == {"engine": "e"}
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


def test_split_summary_accepted_risk_and_escalated_tail():
    from onepass.gate.gate import split_summary

    esc = [False, False, False, True, True]
    s1 = [1, 1, 0, 0, 0]  # System 1 was wrong on both escalated questions and on one accepted one
    fin = [1, 1, 0, 1, 0]  # System 2 fixed one of the two
    lat = [1.0, 1.0, 1.0, 100.0, 300.0]
    usd = [0.001, 0.001, 0.001, 0.1, 0.1]
    s = split_summary(esc, s1, fin, lat, usd, alpha=0.05)
    assert s["n"] == 5 and s["escalate_rate"] == 0.4
    acc = s["accepted"]
    assert acc["n"] == 3 and abs(acc["risk"] - 1 / 3) < 1e-9 and acc["risk_ci_low"] < 1 / 3 < acc["risk_ci_high"]
    e = s["escalated"]
    assert e["n"] == 2 and e["system1_accuracy"] == 0.0 and e["final_accuracy"] == 0.5 and e["mean_ms"] == 200.0
    assert abs(e["usd_share"] - 0.2 / 0.203) < 1e-9
    assert abs(s["escalation_gain_pp"] - 20.0) < 1e-9  # one of five decisions fixed by escalation
    z = split_summary([False] * 3, [1, 1, 1], [1, 1, 1], [1.0] * 3, [0.0] * 3)
    assert z["escalated"]["n"] == 0 and z["escalation_gain_pp"] == 0.0 and z["accepted"]["risk"] == 0.0


def test_specialist_cascade_accounting():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("cascade_specialist", "scripts/cascade_specialist.py")
    cs = importlib.util.module_from_spec(spec)
    sys.modules["cascade_specialist"] = cs
    spec.loader.exec_module(cs)
    gold = [0, 1, 2, 0]
    fin, s1ok, s2ok, esc, lat, usd = cs.specialist_cascade(gold, [0, 0, 2, 1], [0.99, 0.5, 0.95, 0.4], [0, 1, 1, 0], tau=0.9, s1_ms=10, s2_ms=30)
    assert esc == [False, True, False, True]
    assert s1ok == [1, 0, 1, 0] and s2ok == [1, 1, 0, 1] and fin == [1, 1, 1, 1]
    assert lat == [10, 40, 10, 40] and abs(sum(usd) - 100 / 1000 / 3600 * 4.284) < 1e-12
