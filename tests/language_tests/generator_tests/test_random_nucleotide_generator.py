"""Tests for the RandomNucleotideGenerator."""

import copy

import pytest
from proto_tools.transforms.masking import MaskingStrategy

from proto_language.language.core import Segment
from proto_language.language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)


class TestRandomNucleotideGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        assert gen.masking_strategy.num_mutations == 1

    def test_initialization_default(self):
        """Tests default config initialization (random 30%)."""
        config = RandomNucleotideGeneratorConfig()
        gen = RandomNucleotideGenerator(config)
        assert gen.masking_strategy.num_mutations is None
        assert gen.masking_strategy.mask_fraction is None

    def test_assign_and_initialization(self):
        """Tests assign sets segments and that sample mutates an input template."""
        seq_len = 20
        predefined_seq = "A" * seq_len
        segment = Segment(sequence=predefined_seq, sequence_type="rna")
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        gen.assign(segment)

        assert gen._assigned_segments == (segment,)
        assert segment.num_results == 1

        gen.sample()
        assert len(segment.proposal_sequences[0].sequence) == seq_len
        assert all(c in "ACGU" for c in segment.proposal_sequences[0].sequence)

    def test_sample_mutates_sequence(self):
        """Tests the sample method masks and fills one position."""
        seq_len = 25
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        gen._set_program_seed(42)
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        gen.sample()
        mutated_sequence = segment.proposal_sequences[0].sequence

        assert len(mutated_sequence) == seq_len
        # All characters should be valid DNA
        assert all(c in "ACGT" for c in mutated_sequence)

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch of proposals independently."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=5),
        )
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A" * 30, sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment.proposal_sequences]

        # Each sequence should have the right length
        for seq in mutated_sequences:
            assert len(seq) == 30
        # With 5 mutations per sequence, batch should produce diverse results
        assert len(set(mutated_sequences)) > 1

    def test_sample_len_one_sequence(self):
        """Tests that a sequence of length 1 is mutated correctly."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A", sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        gen.sample()
        mutated_char = segment.proposal_sequences[0].sequence

        assert len(mutated_char) == 1
        assert mutated_char in "ACGT"

    def test_num_mutations_parameter(self):
        """Tests that specifying num_mutations masks exactly that many positions."""
        seq_len = 30
        num_mut = 5
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=num_mut),
        )
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.proposal_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.proposal_sequences[0].sequence

        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence, strict=False) if a != b)
        # At most num_mut positions differ (some random replacements may match original)
        assert diff_count <= num_mut
        assert len(mutated_sequence) == seq_len

    def test_substitution_scheme_parameter(self):
        """Tests that substitution_scheme parameter is passed to tool config."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
            substitution_scheme="R",
        )
        gen = RandomNucleotideGenerator(config)
        assert gen.substitution_scheme == "R"

    def test_seed_reproducibility(self):
        """Tests that _set_program_seed produces reproducible results with pre-masked input."""
        # Use pre-masked sequences to bypass position selection randomness
        seq = "ACG_ACG_ACG_ACGT"
        config = RandomNucleotideGeneratorConfig()

        results = []
        for _ in range(2):
            gen = RandomNucleotideGenerator(config)
            gen._set_program_seed(42)
            segment = Segment(sequence=seq, sequence_type="dna")
            gen.assign(segment)
            segment.proposal_sequences = [copy.deepcopy(segment.original_sequence)]
            gen.sample()
            results.append(segment.proposal_sequences[0].sequence)

        assert results[0] == results[1]
        # Verify mask tokens were replaced
        assert "_" not in results[0]


class TestRandomNucleotideGeneratorValidation:
    """Test sequence type validation for RandomNucleotide generator."""

    def test_accepts_dna_segment(self):
        """RandomNucleotide should accept DNA segments."""
        config = RandomNucleotideGeneratorConfig()
        generator = RandomNucleotideGenerator(config)
        segment = Segment(length=50, sequence_type="dna")

        generator.assign(segment)
        assert generator._assigned_segments == (segment,)

    def test_accepts_rna_segment(self):
        """RandomNucleotide should accept RNA segments."""
        config = RandomNucleotideGeneratorConfig()
        generator = RandomNucleotideGenerator(config)
        segment = Segment(length=50, sequence_type="rna")

        generator.assign(segment)
        assert generator._assigned_segments == (segment,)

    def test_rejects_protein_segment(self):
        """RandomNucleotide should reject protein segments."""
        config = RandomNucleotideGeneratorConfig()
        generator = RandomNucleotideGenerator(config)
        segment = Segment(length=50, sequence_type="protein")

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg


class TestRandomNucleotideGeneratorEmptyInit:
    """Test auto-initialization when the segment has no starting sequence."""

    @pytest.mark.parametrize(("seq_type", "alphabet"), [("dna", "ACGT"), ("rna", "ACGU")])
    def test_empty_segment_initialized_randomly(self, seq_type, alphabet):
        """Length-only segment is filled with a random nucleotide sequence on first sample."""
        seq_len = 40
        config = RandomNucleotideGeneratorConfig()
        gen = RandomNucleotideGenerator(config)
        gen._set_program_seed(42)
        segment = Segment(length=seq_len, sequence_type=seq_type)
        gen.assign(segment)

        assert not segment.proposal_sequences[0].sequence
        gen.sample()

        sampled = segment.proposal_sequences[0].sequence
        assert len(sampled) == seq_len
        assert all(c in alphabet for c in sampled)

    def test_empty_segment_with_custom_masking_strategy_ignored_on_init(self):
        """A user-set masking_strategy is bypassed on the init call, then applies normally."""
        seq_len = 30
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=3),
        )
        gen = RandomNucleotideGenerator(config)
        gen._set_program_seed(7)
        segment = Segment(length=seq_len, sequence_type="dna")
        gen.assign(segment)

        gen.sample()
        init_seq = segment.proposal_sequences[0].sequence
        assert len(init_seq) == seq_len
        assert all(c in "ACGT" for c in init_seq)

        gen.sample()
        mutated = segment.proposal_sequences[0].sequence
        diff_count = sum(1 for a, b in zip(init_seq, mutated, strict=True) if a != b)
        assert diff_count <= 3
        assert len(mutated) == seq_len

    def test_empty_segment_batch_all_initialized(self):
        """When all proposals in a batch are empty, each gets its own random init."""
        seq_len = 25
        config = RandomNucleotideGeneratorConfig()
        gen = RandomNucleotideGenerator(config)
        gen._set_program_seed(13)
        segment = Segment(length=seq_len, sequence_type="dna")
        gen.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(4)]

        gen.sample()

        for proposal in segment.proposal_sequences:
            assert len(proposal.sequence) == seq_len
            assert all(c in "ACGT" for c in proposal.sequence)
        assert len({p.sequence for p in segment.proposal_sequences}) > 1

    def test_empty_segment_respects_substitution_scheme(self):
        """Init samples nucleotides from the configured IUPAC pool."""
        seq_len = 50
        config = RandomNucleotideGeneratorConfig(substitution_scheme="R")
        gen = RandomNucleotideGenerator(config)
        gen._set_program_seed(11)
        segment = Segment(length=seq_len, sequence_type="dna")
        gen.assign(segment)

        gen.sample()
        sampled = segment.proposal_sequences[0].sequence
        assert all(c in "AG" for c in sampled)

    def test_empty_segment_init_warns(self, caplog):
        """Init emits a WARNING that mentions the segment label and substitution scheme."""
        config = RandomNucleotideGeneratorConfig(substitution_scheme="Y")
        gen = RandomNucleotideGenerator(config)
        segment = Segment(length=10, sequence_type="dna", label="promoter")
        gen.assign(segment)

        with caplog.at_level("WARNING", logger="proto_language.language.generator.random_nucleotide_generator"):
            gen.sample()

        init_warnings = [
            r for r in caplog.records if r.levelname == "WARNING" and "promoter" in r.message and "Y" in r.message
        ]
        assert init_warnings, f"expected init warning, got: {[r.message for r in caplog.records]}"

    def test_no_warning_on_populated_segment(self, caplog):
        """No init warning is emitted when proposals are already populated."""
        config = RandomNucleotideGeneratorConfig()
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        gen.assign(segment)

        with caplog.at_level("WARNING", logger="proto_language.language.generator.random_nucleotide_generator"):
            gen.sample()

        init_warnings = [r for r in caplog.records if "random init" in r.message]
        assert not init_warnings
