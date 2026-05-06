"""tests/language_tests/generator_tests/test_base_generator.py."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from proto_language.language.core import Generator, Segment, Sequence
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

    def _sample(self) -> None:
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
        assert gen._assigned_segments is None

        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key:
            for seq, seq_type in [("ATCG", "dna"), ("ACGU", "rna"), ("MKKL", "protein")]:
                segment = Segment(sequence=seq, sequence_type=seq_type)
                gen.assign(segment)
                assert gen._assigned_segments == (segment,)
                assert gen.segment is segment

    def test_assign_rejects_ligand_and_incompatible_type(self):
        """Tests that assign rejects ligand segments and incompatible sequence types."""
        gen = ConcreteGenerator()

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            with pytest.raises(ValueError, match="does not support sequence type"):
                gen.assign(Segment(sequence="ATCG", sequence_type="dna"))

    def test_assign_multiple_segments(self):
        """Tests that assign can record multiple target segments."""
        gen = ConcreteGenerator()
        segments = [
            Segment(sequence="MKKL", sequence_type="protein"),
            Segment(sequence="MAAA", sequence_type="protein"),
        ]

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segments)

        assert gen._assigned_segments == tuple(segments)
        assert gen.segment is segments[0]
        assert gen.segments == tuple(segments)

    def test_assign_multiple_segments_requires_tie_compatible_segments(self):
        """Tests that tied segments must describe the same value space."""
        gen = ConcreteGenerator()

        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key:
            with pytest.raises(ValueError, match="different sequence types"):
                gen.assign(
                    [
                        Segment(sequence="ATCG", sequence_type="dna"),
                        Segment(sequence="ACGU", sequence_type="rna"),
                    ]
                )
            with pytest.raises(ValueError, match="different lengths"):
                gen.assign(
                    [
                        Segment(sequence="MKKL", sequence_type="protein"),
                        Segment(sequence="MAAAA", sequence_type="protein"),
                    ]
                )

    def test_assign_rejects_duplicate_segment_instances(self):
        """Reusing one Segment object as multiple tied entries is rejected (defeats tying)."""
        gen = ConcreteGenerator()
        seg = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key, pytest.raises(ValueError, match="duplicate Segment instances"):
            gen.assign([seg, seg])

    def test_assign_rejects_empty_iterable(self):
        """An empty iterable must raise rather than leave the generator in a half-assigned state."""
        gen = ConcreteGenerator()
        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key, pytest.raises(ValueError, match="at least one segment"):
            gen.assign([])
        assert not gen.is_assigned

    def test_assign_rejects_mismatched_valid_chars(self):
        """Tied segments with custom but differing ``valid_chars`` are rejected.

        Without this guard, a primary's mirror would silently overwrite a tied
        segment with characters outside that segment's allowed alphabet.
        """
        gen = ConcreteGenerator()
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key, pytest.raises(ValueError, match="different valid character sets"):
            gen.assign(
                [
                    Segment(sequence="MKKL", sequence_type="protein", valid_chars={"M", "K", "L"}),
                    Segment(sequence="MAAA", sequence_type="protein", valid_chars={"M", "A"}),
                ]
            )

    def test_sample_mirrors_proposals_to_tied_segments(self):
        """``sample()`` deep-copies primary proposals onto every tied segment."""
        from proto_language.language.core import Sequence

        gen = ConcreteGenerator()
        segments = [
            Segment(sequence="MKKL", sequence_type="protein"),
            Segment(sequence="MAAA", sequence_type="protein"),
            Segment(sequence="MGGG", sequence_type="protein"),
        ]
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segments)

        # _sample is a no-op; plant a primary proposal so we observe the mirror.
        segments[0].proposal_sequences = [Sequence(sequence="WWWW", sequence_type="protein")]
        segments[1].proposal_sequences = [Sequence(sequence="ZZZZ", sequence_type="protein")]
        segments[2].proposal_sequences = [Sequence(sequence="QQQQ", sequence_type="protein")]

        gen.sample()

        assert segments[1].proposal_sequences[0].sequence == "WWWW"
        assert segments[2].proposal_sequences[0].sequence == "WWWW"
        # Deep copies, not aliases.
        assert segments[1].proposal_sequences[0] is not segments[0].proposal_sequences[0]
        assert segments[2].proposal_sequences[0] is not segments[0].proposal_sequences[0]

    def test_sample_is_noop_mirror_for_single_segment(self):
        """Single-segment ``sample()`` skips the deepcopy path."""
        from proto_language.language.core import Sequence

        gen = ConcreteGenerator()
        segment = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segment)

        original = Sequence(sequence="WWWW", sequence_type="protein")
        segment.proposal_sequences = [original]
        gen.sample()
        # Same instance — no copying happened.
        assert segment.proposal_sequences[0] is original

    def test_mutation_generator_lazy_init_and_preserves_existing(self):
        """Tests lazy random init for length-only segments and preservation of existing sequences."""
        from proto_tools.transforms.masking import MaskingStrategy

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

        from proto_tools.transforms.masking import MaskingStrategy

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
        from proto_tools.transforms.masking import MaskingStrategy

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


class TestShortSequenceWarning:
    """Tests for the warning emitted from sample() when autoregressive output is shorter than target."""

    GENERATOR_LOGGER = "proto_language.language.core.generator"

    def _assigned(self, category: str = "autoregressive", num_proposals: int = 3) -> tuple[ConcreteGenerator, Segment]:
        gen = ConcreteGenerator()
        segment = Segment(length=50, sequence_type="protein", label="binder")
        p_get, p_key = _patch_registry(_mock_spec(category=category))
        with p_get, p_key:
            gen.assign(segment)
        segment.proposal_sequences = [Sequence(sequence="", sequence_type="protein") for _ in range(num_proposals)]
        return gen, segment

    def _run(self, gen, target_segment, lengths: list[int], caplog) -> list[logging.LogRecord]:
        def fake_sample() -> None:
            for proposal, length in zip(target_segment.proposal_sequences, lengths, strict=True):
                proposal.sequence = "M" * length

        gen._sample = fake_sample  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger=self.GENERATOR_LOGGER):
            gen.sample()
        return [r for r in caplog.records if "candidates shorter than target_length" in r.getMessage()]

    def test_warns_with_all_candidate_lengths(self, caplog):
        gen, segment = self._assigned()
        records = self._run(gen, segment, [50, 30, 40], caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        msg = records[0].getMessage()
        for fragment in (
            "ConcreteGenerator",
            "'binder'",
            "target_length=50 aa",  # protein segment → aa unit
            "candidate #0: 50 aa",  # full-length candidate is shown too
            "candidate #1: 30 aa",
            "candidate #2: 40 aa",
            "end-of-sequence",
        ):
            assert fragment in msg, f"missing {fragment!r} in {msg!r}"

    @pytest.mark.parametrize(
        "category, lengths",
        [
            ("autoregressive", [50, 50, 50]),  # gate passes but no proposals are short
            ("mutation", [30, 30, 30]),  # category gated out
            ("inverse_folding", [30, 30, 30]),  # category gated out
        ],
    )
    def test_no_warning_when_gate_or_length_check_short_circuits(self, caplog, category, lengths):
        gen, segment = self._assigned(category=category)
        assert not self._run(gen, segment, lengths, caplog)

    def test_warning_fires_once_and_mirrors_to_tied_segments(self, caplog):
        gen = ConcreteGenerator()
        primary = Segment(length=50, sequence_type="protein", label="primary")
        tied = Segment(length=50, sequence_type="protein", label="tied")
        p_get, p_key = _patch_registry(_mock_spec(category="autoregressive", supported_types=["protein"]))
        with p_get, p_key:
            gen.assign([primary, tied])
        primary.proposal_sequences = [Sequence(sequence="", sequence_type="protein")]
        assert len(self._run(gen, primary, [30], caplog)) == 1
        assert tied.proposal_sequences[0].sequence == "M" * 30
