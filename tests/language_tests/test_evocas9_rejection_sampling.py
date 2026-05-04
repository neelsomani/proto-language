"""Tests for EvoCas9 rejection-sampling script helpers."""

import csv

from examples.scripts.evocas9_rejection_sampling import (
    _append_filter_log_rows,
    _collect_filter_log_rows,
)
from proto_language.language.core import Segment, Sequence


def test_filter_log_rows_mark_short_circuited_filters_as_skipped(tmp_path):
    """Test proposal diagnostics include failed and skipped filter states."""
    segment = Segment(length=9, sequence_type="dna", label="crispr_locus")
    failed_seq = Sequence("ATGAAATAA", "dna")
    failed_seq._constraints_metadata = {
        "orf_filter": {
            "score": 0.0,
            "data": {"selected_protein_sequence": "MK"},
        },
        "cas9_phmm_filter": {"score": 1.0, "data": {}},
    }
    passed_seq = Sequence("ATGCCCTAA", "dna")
    passed_seq._constraints_metadata = {
        "orf_filter": {"score": 0.0, "data": {"selected_protein_sequence": "MP"}},
        "cas9_phmm_filter": {"score": 0.0, "data": {}},
        "crispr_array_filter": {"score": 0.0, "data": {"crispr_repeat": "GTTCA"}},
    }
    segment.proposal_sequences = [failed_seq, passed_seq]

    rows = _collect_filter_log_rows(
        round_num=1,
        segments=(segment,),
        filter_specs=[
            ("orf_filter", 0.5),
            ("cas9_phmm_filter", 0.5),
            ("crispr_array_filter", 0.5),
        ],
        outcomes=["cas9_phmm_filter", "Not in results"],
        energy_scores=[float("inf"), 0.0],
        temperature=0.5,
        top_k_val=2,
    )

    assert rows[0]["failed_filter"] == "cas9_phmm_filter"
    assert rows[0]["passed_all_filters"] is False
    assert rows[0]["energy_score"] == ""
    assert rows[0]["filter_status_path"] == ("orf_filter:PASS;cas9_phmm_filter:FAIL;crispr_array_filter:SKIPPED")
    assert rows[1]["failed_filter"] is None
    assert rows[1]["passed_all_filters"] is True
    assert rows[1]["accepted_as_result"] is False
    assert rows[1]["outcome"] == "Not in results"

    output_path = tmp_path / "filter_log.tsv"
    _append_filter_log_rows(output_path, rows)
    with open(output_path, newline="") as handle:
        written_rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(written_rows) == 2
    assert written_rows[0]["failed_filter"] == "cas9_phmm_filter"
    assert written_rows[1]["crispr_repeat"] == "GTTCA"
