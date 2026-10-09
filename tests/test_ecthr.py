"""ECtHR case-review records: ten three-way article questions per case, grouped by case."""
import pytest

from onepass.tasks import ecthr


def _rows():
    a = [{"text": ["fact one", "fact two"], "labels": [3]}, {"text": ["other facts"], "labels": []}]
    b = [{"text": ["fact one", "fact two"], "labels": [1, 3]}, {"text": ["other facts"], "labels": [9]}]
    return a, b


def test_records_are_ten_questions_per_case_with_three_way_gold():
    recs = ecthr.records_from_rows(*_rows(), "test")
    assert len(recs) == 20 and len({r.question.qid for r in recs}) == 10
    first = [r for r in recs if r.group_id == "test:0"]
    assert first[0].state == "fact one\nfact two"
    by = {r.meta["article"]: r.gold for r in first}
    assert by["6"] == 2 and by["3"] == 1 and by["2"] == 0  # violated (A), alleged only (B), neither
    second = {r.meta["article"]: r.gold for r in recs if r.group_id == "test:1"}
    assert second["P1-1"] == 1 and sum(second.values()) == 1
    assert all(len(r.question.options) == 3 for r in recs)


def test_violation_without_allegation_still_counts_as_violation():
    a, b = _rows()
    b[0]["labels"] = []  # task B omits an article task A marks violated (2.5% of training cases)
    by = {r.meta["article"]: r.gold for r in ecthr.records_from_rows(a, b, "x") if r.group_id == "x:0"}
    assert by["6"] == 2


def test_mismatched_case_text_is_rejected():
    a, b = _rows()
    b[1]["text"] = ["different"]
    with pytest.raises(ValueError):
        ecthr.records_from_rows(a, b, "test")
