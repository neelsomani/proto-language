"""Tests for the RandomProteinGenerator."""

import copy

import pytest
from proto_tools.transforms.masking import MaskingStrategy

from proto_language.core import Segment
from proto_language.generator import (
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)


class TestRandomProteinGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomProteinGenerator(config)
        assert gen.masking_strategy.num_mutations == 1

    def test_initialization_default(self):
        """Tests default config initialization (random 30%)."""
        config = RandomProteinGeneratorConfig()
        gen = RandomProteinGenerator(config)
        assert gen.masking_strategy.num_mutations is None
        assert gen.masking_strategy.mask_fraction is None

    def test_sample_mutates_sequence(self):
        """Tests the sample method introduces mutations."""
        seq_len = 25
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomProteinGenerator(config)
        gen._set_program_seed(42)
        segment = Segment(sequence="A" * seq_len, sequence_type="protein")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.proposal_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.proposal_sequences[0].sequence

        assert len(mutated_sequence) == seq_len
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence, strict=False) if a != b)
        assert diff_count == 1

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch."""
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=5),
        )
        gen = RandomProteinGenerator(config)
        segment = Segment(sequence="A" * 30, sequence_type="protein")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment.proposal_sequences]

        # Each sequence should have the right length
        for seq in mutated_sequences:
            assert len(seq) == 30
        # With 5 mutations per sequence, batch should produce diverse results
        assert len(set(mutated_sequences)) > 1

    def test_codon_scheme_parameter(self):
        """Tests that codon_scheme parameter is passed to tool config."""
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
            codon_scheme="NNK",
        )
        gen = RandomProteinGenerator(config)
        assert gen.codon_scheme == "NNK"

    def test_seed_reproducibility(self):
        """Tests that _set_program_seed produces reproducible results with pre-masked input."""
        # Use pre-masked sequences to bypass position selection randomness
        seq = "MKK_LVV_GGG_AAA"
        config = RandomProteinGeneratorConfig()

        results = []
        for _ in range(2):
            gen = RandomProteinGenerator(config)
            gen._set_program_seed(42)
            segment = Segment(sequence=seq, sequence_type="protein")
            gen.assign(segment)
            segment.proposal_sequences = [copy.deepcopy(segment.original_sequence)]
            gen.sample()
            results.append(segment.proposal_sequences[0].sequence)

        assert results[0] == results[1]
        # Verify mask tokens were replaced
        assert "_" not in results[0]

    def test_sample_ties_assigned_segments(self):
        """Tests that one generator samples once and ties all assigned segments."""
        config = RandomProteinGeneratorConfig()
        gen = RandomProteinGenerator(config)
        segment1 = Segment(sequence="MKK_LVV_GGG_AAA", sequence_type="protein")
        segment2 = Segment(sequence="MAA_VVV_LLL_GGG", sequence_type="protein")
        gen.assign([segment1, segment2])

        segment1.proposal_sequences = [copy.deepcopy(segment1.original_sequence)]
        segment2.proposal_sequences = [copy.deepcopy(segment2.original_sequence)]
        gen.sample()

        assert gen._assigned_segments == (segment1, segment2)
        assert gen.segments == (segment1, segment2)
        assert "_" not in segment1.proposal_sequences[0].sequence
        assert segment2.proposal_sequences[0].sequence == segment1.proposal_sequences[0].sequence
        assert len(segment1.proposal_sequences[0].sequence) == segment1.sequence_length
        assert len(segment2.proposal_sequences[0].sequence) == segment2.sequence_length


class TestRandomProteinGeneratorValidation:
    """Test sequence type validation for RandomProtein generator."""

    def test_accepts_protein_segment(self):
        """RandomProtein should accept protein segments."""
        config = RandomProteinGeneratorConfig()
        generator = RandomProteinGenerator(config)
        segment = Segment(length=50, sequence_type="protein")

        generator.assign(segment)
        assert generator._assigned_segments == (segment,)

    @pytest.mark.parametrize("seq_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, seq_type):
        """RandomProtein should reject non-protein segments."""
        config = RandomProteinGeneratorConfig()
        generator = RandomProteinGenerator(config)
        segment = Segment(length=50, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg


class TestRandomProteinGeneratorEmptyInit:
    """Test auto-initialization when the segment has no starting sequence."""

    def test_empty_segment_initialized_randomly(self):
        """Length-only segment is filled with a random protein sequence on first sample."""
        seq_len = 40
        config = RandomProteinGeneratorConfig()
        gen = RandomProteinGenerator(config)
        gen._set_program_seed(42)
        segment = Segment(length=seq_len, sequence_type="protein")
        gen.assign(segment)

        assert not segment.proposal_sequences[0].sequence
        gen.sample()

        sampled = segment.proposal_sequences[0].sequence
        assert len(sampled) == seq_len
        assert "_" not in sampled

    def test_empty_segment_with_custom_masking_strategy_ignored_on_init(self):
        """A user-set masking_strategy is bypassed on the init call, then applies normally."""
        seq_len = 30
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=2),
        )
        gen = RandomProteinGenerator(config)
        gen._set_program_seed(7)
        segment = Segment(length=seq_len, sequence_type="protein")
        gen.assign(segment)

        gen.sample()
        init_seq = segment.proposal_sequences[0].sequence
        assert len(init_seq) == seq_len
        assert "_" not in init_seq

        gen.sample()
        mutated = segment.proposal_sequences[0].sequence
        diff_count = sum(1 for a, b in zip(init_seq, mutated, strict=True) if a != b)
        # num_mutations=2 -> at most 2 positions can differ (a replacement may match the original)
        assert diff_count <= 2
        assert len(mutated) == seq_len

    def test_empty_segment_batch_all_initialized(self):
        """When all proposals in a batch are empty, each gets its own random init."""
        seq_len = 25
        config = RandomProteinGeneratorConfig()
        gen = RandomProteinGenerator(config)
        gen._set_program_seed(13)
        segment = Segment(length=seq_len, sequence_type="protein")
        gen.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(4)]

        gen.sample()

        for proposal in segment.proposal_sequences:
            assert len(proposal.sequence) == seq_len
            assert "_" not in proposal.sequence
        assert len({p.sequence for p in segment.proposal_sequences}) > 1

    def test_empty_segment_init_warns(self, caplog):
        """Init emits a WARNING that mentions the segment label and codon scheme."""
        config = RandomProteinGeneratorConfig(codon_scheme="NNK")
        gen = RandomProteinGenerator(config)
        segment = Segment(length=10, sequence_type="protein", label="binder")
        gen.assign(segment)

        with caplog.at_level("WARNING", logger="proto_language.generator.random_protein_generator"):
            gen.sample()

        init_warnings = [
            r for r in caplog.records if r.levelname == "WARNING" and "binder" in r.message and "NNK" in r.message
        ]
        assert init_warnings, f"expected init warning, got: {[r.message for r in caplog.records]}"

    def test_no_warning_on_populated_segment(self, caplog):
        """No init warning is emitted when proposals are already populated."""
        config = RandomProteinGeneratorConfig()
        gen = RandomProteinGenerator(config)
        segment = Segment(sequence="A" * 20, sequence_type="protein")
        gen.assign(segment)

        with caplog.at_level("WARNING", logger="proto_language.generator.random_protein_generator"):
            gen.sample()

        init_warnings = [r for r in caplog.records if "random init" in r.message]
        assert not init_warnings
