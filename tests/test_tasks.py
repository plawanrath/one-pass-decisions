import pytest

from onepass.schema import DecisionRecord, Question, read_jsonl
from onepass.tasks import guardrail, routing, tracegrade
from onepass.tasks.pools import PoolLockedError, PoolSpec, load_pool, lock_pools


def test_routing_records():
    names = ["transfer", "oos", "weather"]
    rows = [{"text": "send money", "intent": 0}, {"text": "asdf", "intent": 1}, {"text": "rain?", "intent": 2}]
    rec = routing.records_from_rows(rows, names, "test", "domain")
    assert rec[0].question.options[rec[0].gold] == "banking"
    assert rec[1].question.options[rec[1].gold] == routing.OOS and rec[1].meta["oos"]
    assert rec[2].question.options[rec[2].gold] == "utility"
    r150 = routing.records_from_rows(rows, names, "test", "intent150")
    assert len(r150[0].question.options) == 151 and r150[0].question.options[r150[0].gold] == "transfer"


def test_guardrail_spml_shared_state():
    rows = [{"System Prompt": "Be a bank bot", "User Prompt": "ignore rules", "Prompt injection": 1, "Degree": 7},
            {"System Prompt": "Be a bank bot", "User Prompt": "balance?", "Prompt injection": 0, "Degree": 0}]
    rec = guardrail.records_from_spml(rows)
    assert len(rec) == 6 and len({r.state for r in rec}) == 1 and len({r.group_id for r in rec}) == 1
    deg = [r for r in rec if r.question.qid == "degree"]
    assert deg[0].question.options[deg[0].gold] == "7" and deg[0].question.kind == "score"
    act = [r for r in rec if r.question.qid == "action"]
    assert act[0].question.options[act[0].gold] == "block" and act[1].question.options[act[1].gold] == "allow"
    assert "ignore rules" in rec[0].question.text and rec[0].gold == 1


def test_guardrail_records():
    rec = guardrail.records_from_injections([{"text": "ignore previous", "label": 1}], "train")
    assert rec[0].gold == 1 and rec[1].question.options[rec[1].gold] == "block" and rec[1].label_source == "programmatic"
    rec = guardrail.records_from_jailbreaks([{"prompt": "hi", "type": "benign"}], "train")
    assert rec[0].gold == 0 and rec[1].question.options[rec[1].gold] == "allow"
    assert rec[0].group_id == rec[1].group_id
    rec = guardrail.records_from_jbhub([{"prompt": "DAN", "jailbreak": True}], "train")
    assert rec[0].question.qid == "jailbreak" and rec[0].gold == 1
    rec = guardrail.records_from_slabs([{"text": "hi", "label": "0"}], "train")
    assert rec[0].question.qid == "injection" and rec[0].gold == 0


def test_tracegrade_records():
    t = tracegrade.Trajectory("t1", "add 2 and 2", [tracegrade.Step(0, "compute", "x = 2 + 2", "4", False),
                                                   tracegrade.Step(1, "clean", "import shutil; shutil.rmtree('/')", "err", True)], "4", "4", True)
    rec = tracegrade.records_from_trajectory(t, silver={0: {"progress": 4, "supervisor": 0}})
    kinds = [(r.question.qid, r.gold, r.label_source) for r in rec]
    assert ("ran", 1, "programmatic") in kinds and ("safe", 0, "programmatic") in kinds
    assert ("progress", 4, "system2-selfconsistency") in kinds
    assert "STEP UNDER REVIEW" in rec[0].state and all(r.group_id == "t1" for r in rec)


def test_pools_lock_and_verify(tmp_path):
    recs = [DecisionRecord("t", f"s{i}", Question("noul", "p", ["false", "true"]), i % 2, "dataset", group_id=f"g{i // 2}") for i in range(40)]
    spec = PoolSpec("t", seed=1, n_dev=6, n_cal=8, n_test=10)
    m = lock_pools(recs, spec, tmp_path / "p")
    assert m["splits"]["test"]["n"] >= 10 and m["splits"]["train"]["n"] > 0
    groups = {s: {r.group_id for r in load_pool(tmp_path / "p", s)} for s in ("train", "dev", "cal", "test")}
    assert not (groups["test"] & groups["train"]) and not (groups["cal"] & groups["dev"])
    with pytest.raises(PoolLockedError):
        lock_pools(recs, spec, tmp_path / "p")
    (tmp_path / "p" / "test.jsonl").write_text("")
    with pytest.raises(RuntimeError):
        load_pool(tmp_path / "p", "test")
    r = next(read_jsonl(tmp_path / "p" / "dev.jsonl"))
    assert isinstance(r.question, Question)
