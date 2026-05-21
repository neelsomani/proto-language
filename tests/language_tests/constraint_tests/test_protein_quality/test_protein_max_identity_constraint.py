"""Tests for protein maximum identity constraint."""

from unittest.mock import patch

from proto_tools import Mmseqs2Hit, Mmseqs2SearchProteinsOutput, Mmseqs2SequenceSearchResult

from proto_language.constraint import ConstraintRegistry, protein_max_identity_constraint
from proto_language.constraint.protein_quality.protein_max_identity_constraint import ProteinMaxIdentityConfig
from proto_language.core import Sequence


def _reference_fasta(tmp_path) -> str:
    """Create a protein reference FASTA fixture."""
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">hit1\nMKTAYIAK\n")
    return str(fasta)


def test_registry_integration():
    """Test that the protein max identity constraint is registered."""
    spec = ConstraintRegistry.get("protein-max-identity")
    assert spec.label == "Protein Max Identity"
    assert "dna" in spec.supported_sequence_types
    assert "protein" in spec.supported_sequence_types


def test_passes_when_top_hit_identity_is_below_threshold(tmp_path):
    """Test that top hits below the maximum identity pass."""
    sequence = Sequence("ATGAAATAA", "dna")
    fasta = _reference_fasta(tmp_path)
    config = ProteinMaxIdentityConfig(mmseqs_db=fasta, reference_fasta=fasta, max_identity=90.0)
    output = Mmseqs2SearchProteinsOutput(
        success=True,
        metadata={},
        results=[
            Mmseqs2SequenceSearchResult(
                query_id="seq_0",
                query_sequence="MKT",
                hits=[Mmseqs2Hit(target_id="hit1", pident=80.0, evalue=1e-10)],
            )
        ],
    )

    with (
        patch(
            "proto_language.constraint.protein_quality.protein_max_identity_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.constraint.protein_quality.protein_max_identity_constraint.run_mmseqs2_search_proteins"
        ) as mock_run,
    ):
        mock_resolve.return_value = [(["MKT"], {"orfipy_orf_count": 1})]
        mock_run.return_value = output
        result = protein_max_identity_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["identity"] == 80.0
    assert result.metadata["top_hit_target_id"] == "hit1"
    assert result.metadata["nearest_hit_seq"] == "MKTAYIAK"


def test_no_hit_passes_when_configured(tmp_path):
    """Test that no-hit proposals can pass as novel sequences."""
    sequence = Sequence("MKT", "protein")
    fasta = _reference_fasta(tmp_path)
    config = ProteinMaxIdentityConfig(mmseqs_db=fasta, reference_fasta=fasta, max_identity=90.0, pass_no_hits=True)
    output = Mmseqs2SearchProteinsOutput(
        success=True,
        metadata={},
        results=[Mmseqs2SequenceSearchResult(query_id="seq_0", query_sequence="MKT", hits=[])],
    )

    with (
        patch(
            "proto_language.constraint.protein_quality.protein_max_identity_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.constraint.protein_quality.protein_max_identity_constraint.run_mmseqs2_search_proteins"
        ) as mock_run,
    ):
        mock_resolve.return_value = [(["MKT"], {})]
        mock_run.return_value = output
        result = protein_max_identity_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["has_mmseqs_hit"] is False
    assert result.metadata["identity"] == 0.0
