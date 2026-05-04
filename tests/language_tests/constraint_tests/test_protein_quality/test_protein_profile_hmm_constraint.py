"""Tests for protein profile-HMM constraint."""

from unittest.mock import patch

from proto_tools import PyHmmsearchOutput
from proto_tools.tools.gene_annotation.pyhmmer.shared_data_models import DomainHit, SequenceHit

from proto_language.language.constraint import ConstraintRegistry, protein_profile_hmm_constraint
from proto_language.language.constraint.protein_quality.protein_profile_hmm_constraint import ProteinProfileHMMConfig
from proto_language.language.core import Sequence


def _sequence_hit(target_name: str = "seq_0", query_name: str = "cas9") -> SequenceHit:
    """Create a PyHMMER sequence hit fixture."""
    return SequenceHit(
        query_name=query_name,
        query_accession="-",
        query_description="-",
        query_idx=0,
        target_name=target_name,
        target_accession="-",
        target_description="-",
        evalue=1e-20,
        score=100.0,
        bias=0.0,
        sum_score=100.0,
        reported=True,
        included=True,
        pvalue=1e-20,
        num_domains=1,
    )


def _domain_hit(target_name: str = "seq_0", query_name: str = "RuvC_1") -> DomainHit:
    """Create a PyHMMER domain hit fixture."""
    return DomainHit(
        query_name=query_name,
        query_accession="-",
        query_description="-",
        query_idx=0,
        target_name=target_name,
        target_accession="-",
        target_description="-",
        hmm_length=100,
        hmm_from=1,
        hmm_to=80,
        target_from=1,
        target_to=80,
        target_length=120,
        c_evalue=1e-20,
        i_evalue=1e-20,
        domain_score=90.0,
        domain_bias=0.0,
        domain_idx=0,
        env_from=1,
        env_to=80,
        envelope_score=90.0,
        domain_included=True,
        domain_reported=True,
        domain_pvalue=1e-20,
    )


def test_registry_integration():
    """Test that the profile-HMM constraint is registered."""
    spec = ConstraintRegistry.get("protein-profile-hmm")
    assert spec.label == "Protein Profile HMM"
    assert "dna" in spec.supported_sequence_types
    assert "protein" in spec.supported_sequence_types


def test_passes_on_any_sequence_level_hmm_hit(tmp_path):
    """Test that any sequence-level hit passes when no profiles are required."""
    hmm_path = tmp_path / "cas9.hmm"
    hmm_path.write_text("HMMER3/f\n")
    sequence = Sequence("ATGAAATAA", "dna")
    config = ProteinProfileHMMConfig(hmm_path=str(hmm_path))
    output = PyHmmsearchOutput(
        success=True,
        metadata={},
        sequence_hits=[_sequence_hit()],
        domain_hits=[],
    )

    with (
        patch(
            "proto_language.language.constraint.protein_quality.protein_profile_hmm_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.language.constraint.protein_quality.protein_profile_hmm_constraint.run_pyhmmer_hmmsearch"
        ) as mock_run,
    ):
        mock_resolve.return_value = [(["MKT"], {"orfipy_orf_count": 1})]
        mock_run.return_value = output
        result = protein_profile_hmm_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["has_profile_hmm_hit"] is True
    assert result.metadata["resolved_protein_sequence"] == "MKT"


def test_required_profiles_are_matched_from_domain_hits(tmp_path):
    """Test that required profile names are matched against domain hits."""
    hmm_path = tmp_path / "domains.hmm"
    hmm_path.write_text("HMMER3/f\n")
    sequence = Sequence("MKT", "protein")
    config = ProteinProfileHMMConfig(
        hmm_path=str(hmm_path),
        required_profiles=["RuvC_1"],
        profile_match_field="query_name",
    )
    output = PyHmmsearchOutput(
        success=True,
        metadata={},
        sequence_hits=[],
        domain_hits=[_domain_hit(query_name="Cas9_RuvC_1")],
    )

    with (
        patch(
            "proto_language.language.constraint.protein_quality.protein_profile_hmm_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.language.constraint.protein_quality.protein_profile_hmm_constraint.run_pyhmmer_hmmsearch"
        ) as mock_run,
    ):
        mock_resolve.return_value = [(["MKT"], {"input_chain_types": ["protein"]})]
        mock_run.return_value = output
        result = protein_profile_hmm_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["profiles_found"] == ["RuvC_1"]
    assert result.metadata["required_profiles"] == ["RuvC_1"]
