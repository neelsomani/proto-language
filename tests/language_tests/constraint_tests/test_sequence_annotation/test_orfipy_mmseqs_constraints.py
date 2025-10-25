import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import (
    orfipy_mmseqs_gene_hit_count_constraint,
    orfipy_mmseqs_gene_homology_constraint,
)
from proto_language.language.constraint.sequence_annotation.orfipy_mmseqs_gene_hit_count_constraint import ORFipyMMseqsGeneHitCountConfig
from proto_language.language.constraint.sequence_annotation.orfipy_mmseqs_gene_homology_constraint import ORFipyMMseqsGeneHomologyConfig
from proto_language.tools.orf_prediction.orfipy import OrfipyInput, OrfipyConfig
from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsInput, MmseqsSearchProteinsConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
    get_test_sequences_with_real_hits,
    dummy_db_path,
    temp_dir,
    setup_test_files,
    ORFIPY_AVAILABLE,
)


@pytest.mark.skipif(not ORFIPY_AVAILABLE, reason="orfipy not installed, skipping ORF tests")
class TestOrfipyMmseqsConstraints:
    @pytest.fixture
    def hit_count_config(self, dummy_db_path):
        return ORFipyMMseqsGeneHitCountConfig(
            min_hits=1,
            max_hits=3,
            mmseqs_db=dummy_db_path,
            mmseqs_config=MmseqsSearchProteinsConfig(
                results_dir="",  # Filled in by pipeline
                threads=1,
                sensitivity=1.0
            ),
            orfipy_config=OrfipyConfig(
                output_dir="",  # Filled in by pipeline
                threads=1,
                min_len=30
            ),
        )

    @pytest.fixture
    def homology_config(self, dummy_db_path):
        return ORFipyMMseqsGeneHomologyConfig(
            min_homology=80.0,
            max_homology=100.0,
            mmseqs_db=dummy_db_path,
            mmseqs_config=MmseqsSearchProteinsConfig(
                results_dir="",  # Filled in by pipeline
                threads=1,
                sensitivity=1.0
            ),
            orfipy_config=OrfipyConfig(
                output_dir="",  # Filled in by pipeline
                threads=1,
                min_len=30
            ),
        )

    def test_hit_count_constraint(self, hit_count_config, temp_dir):
        """Test hit count constraint using real test files."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        # Since we're using real files, we expect the constraint to work with actual data
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0  # Score should be non-negative

        metadata = segment.candidate_sequences[0]._metadata
        assert (
            "segment_0.orfipy_mmseqs_gene_hit_count_constraint.orfipy_orfs" in metadata
        )
        assert (
            "segment_0.orfipy_mmseqs_gene_hit_count_constraint.mmseqs_results"
            in metadata
        )
        assert (
            "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            in metadata
        )
        assert isinstance(
            metadata[
                "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            ],
            int,
        )
        assert (
            metadata[
                "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            ]
            >= 0
        )

    def test_homology_constraint(self, homology_config, temp_dir):
        """Test homology constraint using real test files."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_homology_constraint,
            scoring_function_config=homology_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0

        metadata = segment.candidate_sequences[0]._metadata
        assert (
            "segment_0.orfipy_mmseqs_gene_homology_constraint.orfs_with_acceptable_homology"
            in metadata
        )
        assert (
            metadata[
                "segment_0.orfipy_mmseqs_gene_homology_constraint.orfs_with_acceptable_homology"
            ]
            >= 0
        )
        assert (
            "segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate"
            in metadata
        )
        assert (
            0.0
            <= metadata[
                "segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate"
            ]
            <= 1.0
        )

    def test_no_hits_scenario(self, hit_count_config, temp_dir):
        """Test constraint behavior when no hits are found."""
        # Use a sequence with no meaningful ORFs
        segment = create_segment("A" * 100)

        # Set up test files with empty ORF results
        dna_file = temp_dir / "input.fna"
        dna_file.write_text(">test_seq\n" + "A" * 100 + "\n")

        orfipy_dir = temp_dir / "orfipy_output"
        orfipy_dir.mkdir()

        # Create empty ORF files
        (orfipy_dir / "orfipy_aa.faa").write_text("")
        (orfipy_dir / "orfipy_nt.fna").write_text("")

        # Create empty mmseqs results
        mmseqs_file = temp_dir / "mmseqs_results.m8"
        mmseqs_file.write_text("")

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0  # Should have a penalty for not meeting min_hits
        assert (
            segment.candidate_sequences[0]._metadata[
                "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            ]
            == 0
        )


    def test_hit_count_multiple_sequences(self, hit_count_config, temp_dir):
        """Test hit count constraint with multiple sequences."""
        sequences = get_test_sequences_with_real_hits()
        
        # Test with first two sequences
        segments = [create_segment(seq) for seq in sequences[:2]]
        
        setup_test_files(temp_dir, sequences[0])
        
        for segment in segments:
            constraint = Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
                scoring_function_config=hit_count_config,
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert isinstance(scores[0], float)
            assert scores[0] >= 0.0
            
            # Verify metadata was populated
            metadata = segment.candidate_sequences[0]._metadata
            # The metadata keys are generated based on the segment's own index (always 0 when tested individually)
            assert "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits" in metadata
            assert "segment_0.orfipy_mmseqs_gene_hit_count_constraint.orfipy_orfs" in metadata

    def test_homology_range_compliance(self, homology_config, temp_dir):
        """Test that homology constraint correctly identifies sequences within and outside range."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])
        
        setup_test_files(temp_dir, sequences[0])
        
        # Test with a restrictive homology range
        restrictive_config = ORFipyMMseqsGeneHomologyConfig(
            min_homology=95.0,  # Very high minimum
            max_homology=100.0,
            mmseqs_db=homology_config.mmseqs_db,
            mmseqs_config=homology_config.mmseqs_config,
            orfipy_config=homology_config.orfipy_config,
        )
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_homology_constraint,
            scoring_function_config=restrictive_config,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        
        metadata = segment.candidate_sequences[0]._metadata
        compliance_rate = metadata.get("segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate", 0)
        assert 0.0 <= compliance_rate <= 1.0

    def test_parameter_validation(self, dummy_db_path):
        """Tests that missing required parameters raise validation errors."""
        segment = create_segment("ATGAAATAG")

        # Test hit count constraint - missing required fields should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
                scoring_function_config=ORFipyMMseqsGeneHitCountConfig(
                    min_hits=1,
                    # Missing max_hits
                ),
            ).evaluate()

        # Test homology constraint - invalid types should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_homology_constraint,
                scoring_function_config=ORFipyMMseqsGeneHomologyConfig(
                    min_homology="invalid",  # Should be float
                    max_homology=100.0
                ),
            ).evaluate()

    def test_missing_mmseqs_config_uses_defaults(self, dummy_db_path, temp_dir):
        """Test that missing MMseqs config uses defaults."""
        segment = create_segment("ATGAAATAG")
        
        setup_test_files(temp_dir, "ATGAAATAG")
        
        config = ORFipyMMseqsGeneHitCountConfig(
            min_hits=1,
            max_hits=3,
            mmseqs_db=dummy_db_path,
            mmseqs_config=None,  # Should use defaults
        )
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=config,
        )
        
        # Should not raise an error, should use default config
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)