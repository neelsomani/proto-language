import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
    ConstraintType,
)
from proto_language.language.constraint import (
    orfipy_mmseqs_gene_hit_count_constraint,
)
from proto_language.schemas import ORFipyKwargs, MMseqsKwargs, ESMFoldKwargs
from .test_utils import (
    create_segment,
    create_batched_segment,
)


class TestConstraintConfigNormalization:
    """Test that Constraint class automatically converts dict configs to Pydantic models."""

    def test_esmfold_kwargs_normalization(self):
        """Test that esmfold_kwargs dict is converted to ESMFoldKwargs model."""
        from proto_language.language.constraint import esmfold_plddt_constraint

        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)

        # Test with dict config (what API sends)
        config_with_dict = {
            "n_replications": 1,
            "esmfold_kwargs": {
                "verbose": True,
                "residue_idx_offset": 256,
                "chain_linker": "G" * 10,
            },
        }

        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config_with_dict,
        )

        # Verify the dict was converted to Pydantic model
        assert "esmfold_kwargs" in constraint.scoring_function_config
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, ESMFoldKwargs)
        assert esmfold_kwargs.verbose == True
        assert esmfold_kwargs.residue_idx_offset == 256
        assert esmfold_kwargs.chain_linker == "G" * 10

    def test_orfipy_mmseqs_kwargs_normalization(self):
        """Test that orfipy_kwargs and mmseqs_kwargs dicts are converted to Pydantic models."""
        segment = create_segment("ATGTCGATCGATGTAG", SequenceType.DNA)

        # Create dummy database file for testing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".faa", delete=False) as f:
            f.write(">test_protein\nMVLSPADKTNVK\n")
            dummy_db_path = f.name

        try:
            config_with_dicts = {
                "min_hits": 1,
                "max_hits": 5,
                "orfipy_kwargs": {
                    "threads": 4,
                    "min_len": 30,
                    "max_len": 1000,
                    "start_codons": "ATG,GTG",
                },
                "mmseqs_kwargs": {
                    "database": dummy_db_path,
                    "threads": 4,
                    "sensitivity": 2.0,
                    "only_top_hits": False,
                },
            }

            constraint = Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
                scoring_function_config=config_with_dicts,
            )

            # Verify dicts were converted to Pydantic models
            assert "orfipy_kwargs" in constraint.scoring_function_config
            assert "mmseqs_kwargs" in constraint.scoring_function_config

            orfipy_kwargs = constraint.scoring_function_config["orfipy_kwargs"]
            mmseqs_kwargs = constraint.scoring_function_config["mmseqs_kwargs"]

            assert isinstance(orfipy_kwargs, ORFipyKwargs)
            assert isinstance(mmseqs_kwargs, MMseqsKwargs)

            # Verify values were preserved
            assert orfipy_kwargs.threads == 4
            assert orfipy_kwargs.min_len == 30
            assert orfipy_kwargs.start_codons == "ATG,GTG"

            assert mmseqs_kwargs.database == dummy_db_path
            assert mmseqs_kwargs.threads == 4
            assert mmseqs_kwargs.sensitivity == 2.0
            assert mmseqs_kwargs.only_top_hits == False

        finally:
            # Clean up
            Path(dummy_db_path).unlink(missing_ok=True)

    def test_mixed_config_normalization(self):
        """Test that configs with both regular params and Pydantic kwargs work correctly."""
        from proto_language.language.constraint import esmfold_plddt_constraint

        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)

        config = {
            "n_replications": 2,  # Regular parameter
            "esmfold_kwargs": {  # Should be converted to Pydantic
                "verbose": False,
                "residue_idx_offset": 1024,
            },
        }

        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config,
        )

        # Regular parameter should remain unchanged
        assert constraint.scoring_function_config["n_replications"] == 2

        # Pydantic parameter should be converted
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, ESMFoldKwargs)
        assert esmfold_kwargs.verbose == False
        assert esmfold_kwargs.residue_idx_offset == 1024

    def test_already_pydantic_models_unchanged(self):
        """Test that configs already containing Pydantic models are left unchanged."""
        from proto_language.language.constraint import esmfold_plddt_constraint

        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)

        # Create config with already-instantiated Pydantic model
        esmfold_model = ESMFoldKwargs(verbose=True, residue_idx_offset=512)
        config = {
            "n_replications": 1,
            "esmfold_kwargs": esmfold_model,  # Already a Pydantic model
        }

        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config,
        )

        # Should be the same instance (not converted again)
        assert constraint.scoring_function_config["esmfold_kwargs"] is esmfold_model
        assert isinstance(
            constraint.scoring_function_config["esmfold_kwargs"], ESMFoldKwargs
        )

    def test_empty_config_handling(self):
        """Test that empty configs are handled gracefully."""
        from proto_language.language.constraint import sequence_length_constraint

        segment = create_segment("ATGTCGATCGATGTAG", SequenceType.DNA)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={},
        )

        assert constraint.scoring_function_config == {}

    def test_invalid_pydantic_conversion_fallback(self):
        """Test that invalid Pydantic conversions fall back to dict (backward compatibility)."""
        from proto_language.language.constraint import esmfold_plddt_constraint

        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)

        # Config with invalid ESMFold parameters
        config = {
            "esmfold_kwargs": {
                "invalid_param": "should_cause_error",
                "verbose": "not_a_boolean",  # Invalid type
            }
        }

        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config,
        )

        # Should fall back to dict when Pydantic conversion fails
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, dict)  # Fallback to original dict
        assert esmfold_kwargs["invalid_param"] == "should_cause_error"

    def test_parser_integration(self):
        """Test that the parser creates constraints with properly normalized configs."""
        from api.core.parser import DarwinParser

        # Darwin JSON with protein constraint
        darwin_data = {
            "constructs": [
                {"type": "protein", "segments": [{"id": "protein_segment"}]}
            ],
            "constraints": [
                {
                    "key": "esmfold-plddt",
                    "config": {
                        "n_replications": 1,
                        "esmfold_kwargs": {"verbose": True, "residue_idx_offset": 256},
                    },
                    "targets": ["protein_segment"],
                }
            ],
            "generators": [
                {
                    "key": "uniform-mutation",
                    "config": {"batch_size": 1, "sequence_length": 20},
                    "targets": ["protein_segment"],
                }
            ],
            "optimization": {"method": "mcmc", "num_steps": 1},
        }

        parser = DarwinParser(darwin_data)
        program = parser.parse()

        # Get the constraint that was created
        constraint = program.constraints[0]

        # Verify that the config was normalized
        assert "esmfold_kwargs" in constraint.scoring_function_config
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, ESMFoldKwargs)
        assert esmfold_kwargs.verbose == True
        assert esmfold_kwargs.residue_idx_offset == 256