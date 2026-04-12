"""tests/language_tests/generator_tests/test_base_generator.py."""

from unittest.mock import MagicMock, patch

import pytest

from proto_language.language.core import Generator, Segment
from proto_language.language.generator.generator_registry import (
    GeneratorRegistry,
    GeneratorSpec,
)


# Concrete implementation for testing the abstract base class
class ConcreteGenerator(Generator):
    """Concrete implementation of Generator for testing purposes."""

    def __init__(self) -> None:
        """Minimal implementation of abstract __init__."""
        super().__init__()

    def sample(self) -> None:
        """Dummy sample implementation that does nothing."""


def _mock_spec(category: str = "mutation", supported_types: list[str] | None = None) -> MagicMock:
    """Create a mock GeneratorSpec with the given category."""
    mock = MagicMock(spec=GeneratorSpec)
    mock.supported_sequence_types = supported_types or []
    mock.category = category
    return mock


def _patch_registry(mock_spec: MagicMock):
    """Return a pair of patch objects for GeneratorRegistry.get and get_key."""
    return (
        patch.object(GeneratorRegistry, "get", return_value=mock_spec),
        patch.object(GeneratorRegistry, "get_key", return_value="concrete-generator"),
    )


class TestGeneratorBase:
    """Tests for the base Generator class functionality."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Tests that Generator cannot be directly instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Generator()

    def test_assign_sets_segment_and_allows_all_types_when_empty(self):
        """Tests that assign sets segment and allows any type when supported_sequence_types is empty."""
        gen = ConcreteGenerator()
        assert gen._assigned_segment is None

        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key:
            for seq, seq_type in [("ATCG", "dna"), ("ACGU", "rna"), ("MKKL", "protein")]:
                segment = Segment(sequence=seq, sequence_type=seq_type)
                gen.assign(segment)
                assert gen._assigned_segment is segment

    def test_assign_rejects_ligand_and_incompatible_type(self):
        """Tests that assign rejects ligand segments and incompatible sequence types."""
        gen = ConcreteGenerator()

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            with pytest.raises(ValueError, match="does not support sequence type"):
                gen.assign(Segment(sequence="ATCG", sequence_type="dna"))

    def test_mutation_generator_lazy_init_and_preserves_existing(self):
        """Tests lazy random init for length-only segments and preservation of existing sequences."""
        from proto_tools.tools.masked_models.masking import MaskingStrategy

        from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig

        config = RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))

        # Length-only segment: lazy random init on sample()
        gen = RandomNucleotideGenerator(config)
        segment = Segment(length=20, sequence_type="dna")
        assert not segment.has_original_sequence
        gen.assign(segment)
        assert segment.original_sequence.sequence == ""
        gen.sample()
        assert len(segment.proposal_sequences[0].sequence) == 20
        assert all(c in "ACGT" for c in segment.proposal_sequences[0].sequence)
        assert not segment.has_original_sequence

        # Predefined sequence: preserved after sample
        gen2 = RandomNucleotideGenerator(config)
        segment2 = Segment(sequence="ATCGATCG", sequence_type="dna")
        gen2.assign(segment2)
        gen2.sample()
        assert segment2.has_original_sequence
        assert len(segment2.proposal_sequences[0].sequence) == 8

    def test_mutation_proposals_get_unique_random_sequences(self):
        """Regression: each proposal must get a unique random sequence."""
        import random

        from proto_tools.tools.masked_models.masking import MaskingStrategy

        from proto_language.language.core import Sequence
        from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig

        random.seed(123)
        segment = Segment(length=50, sequence_type="dna")
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        segment.proposal_sequences = [Sequence(sequence="", sequence_type="dna") for _ in range(5)]
        gen._validate_generator()

        sequences = [s.sequence for s in segment.proposal_sequences]
        assert all(len(s) == 50 for s in sequences)
        assert len(set(sequences)) > 1, "All proposals got the same random sequence"

    def test_autoregressive_generator_no_random_init(self):
        """Tests that autoregressive generators don't initialize random sequences."""
        gen = ConcreteGenerator()
        segment = Segment(length=20, sequence_type="dna")

        p_get, p_key = _patch_registry(_mock_spec(category="autoregressive"))
        with p_get, p_key:
            gen.assign(segment)
        assert segment.original_sequence.sequence == ""

    def test_validate_generator_empty_proposal_pool_raises(self):
        """Tests that _validate_generator raises on empty proposal_sequences."""
        from proto_tools.tools.masked_models.masking import MaskingStrategy

        from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment = Segment(sequence="ATCG", sequence_type="dna")
        gen.assign(segment)
        segment.proposal_sequences = []

        with pytest.raises(RuntimeError, match="empty proposal_sequences pool"):
            gen._validate_generator()


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry functionality."""

    def test_unknown_key_raises(self):
        """Tests that get_key and create raise for unregistered generators."""
        with pytest.raises(ValueError, match="is not registered"):
            GeneratorRegistry.get_key(ConcreteGenerator())
        with pytest.raises(ValueError, match="Unknown generator"):
            GeneratorRegistry.create("nonexistent-generator", {})

    def test_list_all_returns_valid_specs(self):
        """Tests that all specs have required fields."""
        all_specs = GeneratorRegistry.list_all()
        assert len(all_specs) > 0

        for spec in all_specs:
            assert isinstance(spec, GeneratorSpec)
            for attr in ("key", "label", "description", "category", "uses_gpu", "supported_sequence_types"):
                assert hasattr(spec, attr)
