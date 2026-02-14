"""Tests for proto_language.utils.export module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from proto_language.utils.export import (
    flatten_constraints,
    flatten_constructs,
    flatten_optimization,
    flatten_sequences,
    to_csv,
    to_json,
    to_tsv,
    write_export,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_batch_results():
    """Sample batch_results with 2 batches, 1 construct, 2 segments.

    - promoter: gc_content_constraint + length_constraint
    - cds: gc_content_constraint only
    """
    return {
        "batch_results": [
            {
                "batch_idx": 0,
                "energy_score": 0.5,
                "constructs": [
                    {
                        "label": "construct_0",
                        "type": "dna",
                        "segments": [
                            {
                                "label": "promoter",
                                "sequence": "ATCGATCG",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.1,
                                        "weight": 1.0,
                                        "weighted_score": 0.1,
                                        "data": {"gc_content": 50.0},
                                    },
                                    "length_constraint": {
                                        "score": 0.0,
                                        "weight": 1.0,
                                        "weighted_score": 0.0,
                                        "data": {"length": 8},
                                    },
                                },
                                "metadata": {"source": "synthetic"},
                            },
                            {
                                "label": "cds",
                                "sequence": "GCTAGCTA",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.05,
                                        "weight": 1.0,
                                        "weighted_score": 0.05,
                                        "data": {"gc_content": 52.0},
                                    },
                                },
                                "metadata": {},
                            },
                        ],
                    },
                ],
            },
            {
                "batch_idx": 1,
                "energy_score": 0.3,
                "constructs": [
                    {
                        "label": "construct_0",
                        "type": "dna",
                        "segments": [
                            {
                                "label": "promoter",
                                "sequence": "TTAATTAA",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.2,
                                        "weight": 1.0,
                                        "weighted_score": 0.2,
                                        "data": {"gc_content": 25.0},
                                    },
                                    "length_constraint": {
                                        "score": 0.0,
                                        "weight": 1.0,
                                        "weighted_score": 0.0,
                                        "data": {"length": 8},
                                    },
                                },
                                "metadata": {"source": "synthetic"},
                            },
                            {
                                "label": "cds",
                                "sequence": "CCGGCCGG",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.01,
                                        "weight": 1.0,
                                        "weighted_score": 0.01,
                                        "data": {"gc_content": 75.0},
                                    },
                                },
                                "metadata": {},
                            },
                        ],
                    },
                ],
            },
        ],
        "best_batch_idx": 1,
    }


@pytest.fixture
def sample_history():
    """Sample optimizer history using standardized batch_results format."""
    return [
        {
            "time_step": 0,
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.8,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "AAAA",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.5,
                                            "weight": 1.0,
                                            "weighted_score": 0.5,
                                            "data": {"gc_content": 0.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
                {
                    "batch_idx": 1,
                    "energy_score": 0.9,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "TTTT",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.5,
                                            "weight": 1.0,
                                            "weighted_score": 0.5,
                                            "data": {"gc_content": 0.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        },
        {
            "time_step": 10,
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "ATCG",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.2,
                                            "weight": 1.0,
                                            "weighted_score": 0.2,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
                {
                    "batch_idx": 1,
                    "energy_score": 0.6,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "GCTA",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.3,
                                            "weight": 1.0,
                                            "weighted_score": 0.3,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        },
    ]


# =============================================================================
# Test flatten_sequences
# =============================================================================


class TestFlattenSequences:
    """Tests for flatten_sequences: one row per (batch_idx, construct, segment)."""

    def test_row_count(self, sample_batch_results):
        """2 batches x 1 construct x 2 segments = 4 rows."""
        rows = flatten_sequences(sample_batch_results)
        assert len(rows) == 4

    def test_fixed_columns(self, sample_batch_results):
        """Every row has batch_idx, energy_score, construct, segment, sequence."""
        rows = flatten_sequences(sample_batch_results)
        for row in rows:
            assert "batch_idx" in row
            assert "energy_score" in row
            assert "construct" in row
            assert "segment" in row
            assert "sequence" in row

    def test_constraint_columns_present(self, sample_batch_results):
        """Constraint fields use {label}.{field} namespacing."""
        rows = flatten_sequences(sample_batch_results)
        promoter_row = [r for r in rows if r["segment"] == "promoter" and r["batch_idx"] == 0][0]

        # All constraint fields present
        assert promoter_row["gc_content_constraint.score"] == 0.1
        assert promoter_row["gc_content_constraint.weight"] == 1.0
        assert promoter_row["gc_content_constraint.weighted_score"] == 0.1
        assert promoter_row["gc_content_constraint.gc_content"] == 50.0
        assert promoter_row["length_constraint.score"] == 0.0
        assert promoter_row["length_constraint.length"] == 8

    def test_metadata_prefix(self, sample_batch_results):
        """User metadata uses metadata.{key} prefix."""
        rows = flatten_sequences(sample_batch_results)
        promoter_row = [r for r in rows if r["segment"] == "promoter" and r["batch_idx"] == 0][0]
        assert promoter_row["metadata.source"] == "synthetic"

    def test_correct_values(self, sample_batch_results):
        """Spot-check specific values."""
        rows = flatten_sequences(sample_batch_results)
        cds_batch1 = [r for r in rows if r["segment"] == "cds" and r["batch_idx"] == 1][0]
        assert cds_batch1["sequence"] == "CCGGCCGG"
        assert cds_batch1["energy_score"] == 0.3
        assert cds_batch1["gc_content_constraint.gc_content"] == 75.0

    def test_empty_results(self):
        """Handles empty batch_results."""
        assert flatten_sequences({"batch_results": []}) == []


# =============================================================================
# Test flatten_constraints
# =============================================================================


class TestFlattenConstraints:
    """Tests for flatten_constraints: one row per (batch, construct, segment, constraint)."""

    def test_row_count(self, sample_batch_results):
        """batch0: promoter(2) + cds(1) = 3; batch1: same = 3; total = 6."""
        rows = flatten_constraints(sample_batch_results)
        assert len(rows) == 6

    def test_fixed_columns(self, sample_batch_results):
        """Every row has constraint identifier columns + standard metrics."""
        rows = flatten_constraints(sample_batch_results)
        for row in rows:
            assert "batch_idx" in row
            assert "construct" in row
            assert "segment" in row
            assert "constraint" in row
            assert "score" in row
            assert "weight" in row
            assert "weighted_score" in row

    def test_custom_data_unprefixed(self, sample_batch_results):
        """Custom data fields are un-prefixed (one constraint per row)."""
        rows = flatten_constraints(sample_batch_results)
        gc_row = [r for r in rows if r["constraint"] == "gc_content_constraint" and r["batch_idx"] == 0 and r["segment"] == "promoter"][0]
        assert gc_row["gc_content"] == 50.0
        assert gc_row["score"] == 0.1

    def test_multi_segment_info(self):
        """Multi-segment constraints include input_segments and position_in_inputs."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "protein_a",
                                    "sequence": "MVLS",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": ["c0.protein_a", "c0.protein_b"],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }
        rows = flatten_constraints(batch_results)
        assert len(rows) == 1
        assert rows[0]["input_segments"] == ["c0.protein_a", "c0.protein_b"]
        assert rows[0]["position_in_inputs"] == 0
        assert rows[0]["binding_energy"] == -5.0

    def test_empty_results(self):
        """Handles empty batch_results."""
        assert flatten_constraints({"batch_results": []}) == []


# =============================================================================
# Test flatten_constructs
# =============================================================================


class TestFlattenConstructs:
    """Tests for flatten_constructs: one row per (batch_idx, construct)."""

    def test_row_count(self, sample_batch_results):
        """2 batches x 1 construct = 2 rows."""
        rows = flatten_constructs(sample_batch_results)
        assert len(rows) == 2

    def test_full_sequence(self, sample_batch_results):
        """full_sequence is concatenation of all segment sequences."""
        rows = flatten_constructs(sample_batch_results)
        assert rows[0]["full_sequence"] == "ATCGATCGGCTAGCTA"
        assert rows[1]["full_sequence"] == "TTAATTAACCGGCCGG"

    def test_per_segment_columns(self, sample_batch_results):
        """Per-segment data uses {segment}.{field} prefix."""
        rows = flatten_constructs(sample_batch_results)
        row = rows[0]
        assert row["promoter.sequence"] == "ATCGATCG"
        assert row["cds.sequence"] == "GCTAGCTA"
        assert row["promoter.gc_content_constraint.score"] == 0.1
        assert row["promoter.gc_content_constraint.gc_content"] == 50.0
        assert row["cds.gc_content_constraint.gc_content"] == 52.0

    def test_per_segment_metadata(self, sample_batch_results):
        """Per-segment metadata uses {segment}.metadata.{key} prefix."""
        rows = flatten_constructs(sample_batch_results)
        assert rows[0]["promoter.metadata.source"] == "synthetic"

    def test_empty_results(self):
        """Handles empty batch_results."""
        assert flatten_constructs({"batch_results": []}) == []


# =============================================================================
# Test flatten_optimization
# =============================================================================


class TestFlattenOptimization:
    """Tests for flatten_optimization: one row per (timepoint, batch_idx)."""

    def test_row_count(self, sample_history):
        """2 timepoints x 2 batches = 4 rows."""
        rows = flatten_optimization(sample_history)
        assert len(rows) == 4

    def test_fixed_columns(self, sample_history):
        """Every row has timepoint, batch_idx, energy_score."""
        rows = flatten_optimization(sample_history)
        for row in rows:
            assert "timepoint" in row
            assert "batch_idx" in row
            assert "energy_score" in row

    def test_per_segment_sequences(self, sample_history):
        """Per-segment data uses {segment}.sequence columns."""
        rows = flatten_optimization(sample_history)
        # First timepoint, batch 0
        t0_b0 = [r for r in rows if r["timepoint"] == 0 and r["batch_idx"] == 0][0]
        assert t0_b0["promoter.sequence"] == "AAAA"
        assert t0_b0["energy_score"] == 0.8

    def test_per_segment_constraint_scores(self, sample_history):
        """Constraint scores use {segment}.{constraint}.{field} prefix."""
        rows = flatten_optimization(sample_history)
        t10_b0 = [r for r in rows if r["timepoint"] == 10 and r["batch_idx"] == 0][0]
        assert t10_b0["promoter.gc_constraint.score"] == 0.2
        assert t10_b0["promoter.gc_constraint.gc_content"] == 50.0

    def test_batch_energy_scores(self, sample_history):
        """Each batch member has its own energy score."""
        rows = flatten_optimization(sample_history)
        t0_b1 = [r for r in rows if r["timepoint"] == 0 and r["batch_idx"] == 1][0]
        assert t0_b1["energy_score"] == 0.9

    def test_empty_history(self):
        """Handles empty history."""
        assert flatten_optimization([]) == []


# =============================================================================
# Test Format Writers
# =============================================================================


class TestFormatWriters:
    """Tests for format writer functions."""

    @pytest.fixture
    def sample_rows(self):
        """Simple rows for testing format writers."""
        return [
            {"name": "A", "value": 1, "score": 0.5},
            {"name": "B", "value": 2, "score": 0.8},
        ]

    def test_to_csv_returns_string(self, sample_rows):
        """Test to_csv returns valid CSV string."""
        result = to_csv(sample_rows)

        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows
        assert "name,value,score" in lines[0]

    def test_to_csv_writes_to_path(self, sample_rows):
        """Test to_csv writes to file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            to_csv(sample_rows, path)

            assert path.exists()
            content = path.read_text()
            assert "name,value,score" in content

    def test_to_tsv_uses_tabs(self, sample_rows):
        """Test to_tsv uses tab delimiters."""
        result = to_tsv(sample_rows)

        lines = result.strip().split("\n")
        assert "\t" in lines[0]
        assert "," not in lines[0]

    def test_to_json_returns_valid_json(self, sample_rows):
        """Test to_json returns valid JSON."""
        result = to_json(sample_rows)

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "A"

    def test_to_json_writes_to_path(self, sample_rows):
        """Test to_json writes to file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            to_json(sample_rows, path)

            assert path.exists()
            content = json.loads(path.read_text())
            assert len(content) == 2

    def test_empty_rows(self):
        """Test format writers handle empty rows."""
        assert to_csv([]) == ""
        assert to_tsv([]) == ""
        assert to_json([]) == "[]"


class TestWriteExport:
    """Tests for write_export function."""

    @pytest.fixture
    def sample_rows(self):
        return [{"a": 1, "b": 2}]

    def test_csv_format(self, sample_rows):
        """Test write_export with csv format."""
        result = write_export(sample_rows, "csv")
        assert "a,b" in result

    def test_tsv_format(self, sample_rows):
        """Test write_export with tsv format."""
        result = write_export(sample_rows, "tsv")
        assert "a\tb" in result

    def test_json_format(self, sample_rows):
        """Test write_export with json format."""
        result = write_export(sample_rows, "json")
        parsed = json.loads(result)
        assert parsed[0]["a"] == 1

    def test_xlsx_requires_path(self, sample_rows):
        """Test xlsx format raises error without path."""
        with pytest.raises(ValueError, match="xlsx format requires"):
            write_export(sample_rows, "xlsx")

    def test_invalid_format(self, sample_rows):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            write_export(sample_rows, "invalid")


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_missing_constraints(self):
        """Segments with no constraints produce rows with only fixed columns."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.0,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "dna",
                            "segments": [
                                {"label": "s0", "sequence": "ATCG", "constraints": {}, "metadata": {}},
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }

        seq_rows = flatten_sequences(batch_results)
        assert len(seq_rows) == 1
        assert seq_rows[0]["sequence"] == "ATCG"

        constraint_rows = flatten_constraints(batch_results)
        assert len(constraint_rows) == 0  # No constraints = no rows

    def test_heterogeneous_constraints(self):
        """Segments with different constraints produce union of columns in CSV."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "s0",
                                    "sequence": "ATCG",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                                {
                                    "label": "s1",
                                    "sequence": "GCTA",
                                    "constraints": {
                                        "length_constraint": {
                                            "score": 0.0,
                                            "weight": 1.0,
                                            "weighted_score": 0.0,
                                            "data": {"length": 4},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }

        rows = flatten_sequences(batch_results)
        csv_output = to_csv(rows)

        # CSV should have columns from both constraints
        assert "gc_constraint.gc_content" in csv_output
        assert "length_constraint.length" in csv_output

    def test_multi_segment_constraint_in_sequences(self):
        """Multi-segment constraints include input_segments in sequences table."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "protein_a",
                                    "sequence": "MVLS",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": ["c0.protein_a", "c0.protein_b"],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                                {
                                    "label": "protein_b",
                                    "sequence": "KAAW",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": ["c0.protein_a", "c0.protein_b"],
                                            "position_in_inputs": 1,
                                            "data": {"interface_contacts": 12},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }

        rows = flatten_sequences(batch_results)
        assert len(rows) == 2

        csv_output = to_csv(rows)
        assert "interaction.score" in csv_output
        assert "interaction.input_segments" in csv_output

    def test_constructs_multi_segment_constraint(self):
        """Constructs table with multi-segment constraint uses full prefix."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "protein_a",
                                    "sequence": "MVLS",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": ["c0.protein_a", "c0.protein_b"],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }

        rows = flatten_constructs(batch_results)
        assert len(rows) == 1
        assert rows[0]["protein_a.interaction.score"] == 0.1
        assert rows[0]["protein_a.interaction.binding_energy"] == -5.0

    def test_csv_round_trip(self, sample_batch_results):
        """Flatten → CSV → parse produces correct data."""
        rows = flatten_sequences(sample_batch_results)
        csv_str = to_csv(rows)

        # Parse CSV back
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(csv_str))
        parsed_rows = list(reader)
        assert len(parsed_rows) == 4
        assert parsed_rows[0]["sequence"] == "ATCGATCG"


# =============================================================================
# Test Filtering
# =============================================================================


class TestFiltering:
    """Tests for segment/constraint filtering on flatten functions."""

    def test_sequences_filter_segment(self, sample_batch_results):
        """Only include rows for the specified segment."""
        rows = flatten_sequences(sample_batch_results, segments={"promoter"})
        assert len(rows) == 2  # 2 batches x 1 segment
        assert all(r["segment"] == "promoter" for r in rows)

    def test_sequences_filter_segment_multiple(self, sample_batch_results):
        """Filter by multiple segments returns rows for all specified."""
        rows = flatten_sequences(
            sample_batch_results, segments={"promoter", "cds"}
        )
        assert len(rows) == 4  # All segments included

    def test_constraints_filter_segment(self, sample_batch_results):
        """Only include constraint rows for the specified segment."""
        rows = flatten_constraints(sample_batch_results, segments={"cds"})
        # cds has 1 constraint x 2 batches = 2 rows
        assert len(rows) == 2
        assert all(r["segment"] == "cds" for r in rows)

    def test_constraints_filter_constraint(self, sample_batch_results):
        """Only include rows for the specified constraint label."""
        rows = flatten_constraints(
            sample_batch_results, constraints={"length_constraint"}
        )
        # Only promoter has length_constraint, 2 batches = 2 rows
        assert len(rows) == 2
        assert all(r["constraint"] == "length_constraint" for r in rows)

    def test_constraints_filter_both(self, sample_batch_results):
        """Filter by both segment and constraint simultaneously."""
        rows = flatten_constraints(
            sample_batch_results,
            segments={"promoter"},
            constraints={"gc_content_constraint"},
        )
        # promoter's gc_content_constraint x 2 batches = 2 rows
        assert len(rows) == 2
        assert all(r["segment"] == "promoter" for r in rows)
        assert all(r["constraint"] == "gc_content_constraint" for r in rows)

    def test_constructs_filter_segment(self, sample_batch_results):
        """Only include per-segment columns for the specified segment."""
        rows = flatten_constructs(sample_batch_results, segments={"promoter"})
        assert len(rows) == 2  # Row count unchanged (one row per construct)
        row = rows[0]
        assert "promoter.sequence" in row
        assert "cds.sequence" not in row
        # full_sequence still includes all segments
        assert row["full_sequence"] == "ATCGATCGGCTAGCTA"

    def test_optimization_filter_segment(self, sample_history):
        """Only include per-segment columns for the specified segment."""
        rows = flatten_optimization(sample_history, segments={"promoter"})
        assert len(rows) == 4  # Row count unchanged
        assert "promoter.sequence" in rows[0]
        # No other segment columns should appear
        non_fixed_keys = {
            k for r in rows for k in r if not k.startswith(("timepoint", "batch_idx", "energy_score"))
        }
        assert all(k.startswith("promoter.") for k in non_fixed_keys)

    def test_filter_nonexistent_segment(self, sample_batch_results):
        """Filtering by a segment that doesn't exist returns no segment rows."""
        rows = flatten_sequences(
            sample_batch_results, segments={"nonexistent"}
        )
        assert len(rows) == 0

    def test_filter_nonexistent_constraint(self, sample_batch_results):
        """Filtering by a constraint that doesn't exist returns no rows."""
        rows = flatten_constraints(
            sample_batch_results, constraints={"nonexistent"}
        )
        assert len(rows) == 0

    def test_sequences_filter_batch_idx(self, sample_batch_results):
        """Only include rows for specified batch indices."""
        rows = flatten_sequences(sample_batch_results, batch_indices={0})
        assert len(rows) == 2  # 1 batch x 2 segments
        assert all(r["batch_idx"] == 0 for r in rows)

    def test_constraints_filter_batch_idx(self, sample_batch_results):
        """Only include constraint rows for specified batch indices."""
        rows = flatten_constraints(sample_batch_results, batch_indices={1})
        # batch 1: promoter(2 constraints) + cds(1) = 3 rows
        assert len(rows) == 3
        assert all(r["batch_idx"] == 1 for r in rows)

    def test_constructs_filter_batch_idx(self, sample_batch_results):
        """Only include construct rows for specified batch indices."""
        rows = flatten_constructs(sample_batch_results, batch_indices={0})
        assert len(rows) == 1
        assert rows[0]["batch_idx"] == 0

    def test_optimization_filter_batch_idx(self, sample_history):
        """Only include optimization rows for specified batch indices."""
        rows = flatten_optimization(sample_history, batch_indices={0})
        assert len(rows) == 2  # 2 timepoints x 1 batch
        assert all(r["batch_idx"] == 0 for r in rows)

    def test_combined_segment_and_batch_filter(self, sample_batch_results):
        """Filter by both segment and batch index."""
        rows = flatten_sequences(
            sample_batch_results,
            segments={"promoter"},
            batch_indices={1},
        )
        assert len(rows) == 1
        assert rows[0]["segment"] == "promoter"
        assert rows[0]["batch_idx"] == 1
        assert rows[0]["sequence"] == "TTAATTAA"
