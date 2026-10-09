"""ContractNLI records: one per (contract, hypothesis), grouped by contract, gold mapped to the three options."""
from onepass.tasks.contractnli import OPTIONS, records_from_split

DOC = {"id": 7, "text": "The Recipient shall return all materials. Nothing else.", "spans": [[0, 41], [42, 55]],
       "annotation_sets": [{"annotations": {"nda-2": {"choice": "NotMentioned", "spans": []},
                                            "nda-1": {"choice": "Entailment", "spans": [0]},
                                            "nda-10": {"choice": "Contradiction", "spans": [1]}}}]}
LABELS = {"nda-1": {"short_description": "Return", "hypothesis": "Receiving Party shall return information."},
          "nda-2": {"short_description": "Other", "hypothesis": "Something else."},
          "nda-10": {"short_description": "Ten", "hypothesis": "A tenth thing."}}


def test_records_one_per_hypothesis_grouped_by_contract():
    recs = records_from_split({"documents": [DOC], "labels": LABELS}, "test")
    assert [r.question.qid for r in recs] == ["nda-1", "nda-2", "nda-10"]  # numeric hypothesis order
    assert [OPTIONS[r.gold] for r in recs] == ["entailed", "not mentioned", "contradicted"]
    assert {r.group_id for r in recs} == {"test:7"} and all(r.state == DOC["text"] for r in recs)
    assert recs[0].meta["evidence_chars"] == [[0, 41]] and recs[1].meta["evidence_chars"] == []
    assert recs[0].question.kind == "choice" and recs[0].question.options == OPTIONS
