import copy

import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    Evo1Generator,
    Evo1GeneratorConfig,
)


@pytest.mark.uses_gpu
class TestEvo1Generator:
    def test_evo1_generation(self):
        """Test generation: custom params, batching, and prompt replication."""
        # Single prompt replicated to 3 candidates with custom params and batching
        prompts = ["ATCG"]
        num_tokens = 50
        expected_length = len(prompts[0]) + num_tokens
        config = Evo1GeneratorConfig(
            prompts=prompts,
            temperature=0.8,
            top_k=10,
            batch_size=2,
        )
        gen = Evo1Generator(config)

        assert gen.temperature == 0.8
        assert gen.top_k == 10
        assert gen.batch_size == 2

        segment = Segment(length=expected_length, sequence_type="dna")
        segment.candidate_sequences = [
            copy.deepcopy(segment.original_sequence) for _ in range(3)
        ]
        gen.assign(segment)
        assert gen._assigned_segment is segment

        gen.sample()

        for i in range(3):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[0])
            assert segment.candidate_sequences[i].sequence_type == "dna"

    def test_evo1_prompt_mismatch_raises(self):
        """3 prompts with 2 candidates should raise ValueError."""
        prompts = ["ATCG", "GGCC", "TTAA"]
        config = Evo1GeneratorConfig(prompts=prompts)
        gen = Evo1Generator(config)

        segment = Segment(length=104, sequence_type="dna")
        segment.candidate_sequences = [
            copy.deepcopy(segment.original_sequence) for _ in range(2)
        ]
        gen.assign(segment)

        with pytest.raises(ValueError, match="must either be 1"):
            gen.sample()


class TestEvo1GeneratorValidation:
    """Test sequence type validation for Evo1 generator."""

    def test_valid_dna_assignment(self):
        """Evo1 should accept DNA segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="dna")

        # Should not raise
        gen.assign(segment)
        assert gen._assigned_segment is segment

    def test_rejects_protein_segment(self):
        """Evo1 should reject PROTEIN segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="protein")

        with pytest.raises(ValueError) as exc_info:
            gen.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert "protein" in error_msg.lower()
        assert "dna" in error_msg.lower()

    def test_rejects_rna_segment(self):
        """Evo1 should reject RNA segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="rna")

        with pytest.raises(ValueError) as exc_info:
            gen.assign(segment)

        assert "does not support sequence type" in str(exc_info.value)
        assert "rna" in str(exc_info.value).lower()

    def test_batch_size_config(self):
        """Test that batch_size parameter is properly set in config and generator."""
        config = Evo1GeneratorConfig(prompts="ATGC", batch_size=5)
        gen = Evo1Generator(config)

        assert config.batch_size == 5
        assert gen.batch_size == 5

    def test_prompts_unequal_length_raises(self):
        """Prompts with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            Evo1GeneratorConfig(prompts=["ATCG", "AT"])
