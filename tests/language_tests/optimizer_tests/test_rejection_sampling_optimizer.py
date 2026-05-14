"""Minimal tests verifying core behavior of the RejectionSamplingOptimizer."""

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from proto_tools.transforms.masking import MaskingStrategy
from pydantic import BaseModel

from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    GeneratorInputType,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.language.generator.generator_registry import (
    GeneratorRegistry,
    GeneratorSpec,
)
from proto_language.language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig
from proto_language.language.optimizer.rejection_sampling_optimizer import DID_NOT_ENTER_TOP_K


class _NoOpGenerator(Generator):
    """Generator that does not mutate sequences. Used for testing optimizer logic."""

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self):
        super().__init__()

    def _sample(self) -> None:
        self._validate_generator()


class _BatchSizeConfig(BaseModel):
    batch_size: int | None = None


def _batch_size_test_constraint(input_sequences, config):
    return []


_batch_size_test_constraint._constraint_supported_sequence_types = ["dna"]
_batch_size_test_constraint._constraint_num_input_sequences_per_tuple = 1


def _make_noop_generator(segment):
    """Create a no-op generator assigned to a segment, with registry mocking."""
    gen = _NoOpGenerator()
    mock_spec = MagicMock(spec=GeneratorSpec)
    mock_spec.supported_sequence_types = []
    mock_spec.category = "mutation"
    with patch.object(GeneratorRegistry, "get", return_value=mock_spec):
        with patch.object(GeneratorRegistry, "get_key", return_value="noop"):
            gen.assign(segment)
    return gen


def _make_batch_size_constraint(segment, *, function_config=None, backward_config=None):
    """Create a constraint whose configs can carry batch_size for inference tests."""
    return Constraint(
        inputs=[segment],
        function=_batch_size_test_constraint,
        function_config=function_config,
        backward_config=backward_config,
    )


class TestRejectionSamplingProposalBatchSizeInference:
    """Test auto-inference of proposal_batch_size from batched components."""

    def test_infers_from_generator_batch_size(self):
        segment = Segment(sequence="ATCG", sequence_type="dna")
        gen = _make_noop_generator(segment)
        gen.batch_size = 7

        assert (
            RejectionSamplingOptimizer._resolve_proposal_batch_size(
                generators=[gen],
                constraints=[],
                num_samples=20,
                configured=None,
            )
            == 7
        )

    def test_infers_from_constraint_config_batch_size(self):
        segment = Segment(sequence="ATCG", sequence_type="dna")
        constraint = _make_batch_size_constraint(segment, function_config={"batch_size": 5})

        assert (
            RejectionSamplingOptimizer._resolve_proposal_batch_size(
                generators=[],
                constraints=[constraint],
                num_samples=20,
                configured=None,
            )
            == 5
        )

    def test_uses_largest_component_batch_size(self):
        segment = Segment(sequence="ATCG", sequence_type="dna")
        gen = _make_noop_generator(segment)
        gen.batch_size = 3
        function_constraint = _make_batch_size_constraint(segment, function_config={"batch_size": 6})
        backward_constraint = _make_batch_size_constraint(segment, backward_config=_BatchSizeConfig(batch_size=9))

        assert (
            RejectionSamplingOptimizer._resolve_proposal_batch_size(
                generators=[gen],
                constraints=[function_constraint, backward_constraint],
                num_samples=20,
                configured=None,
            )
            == 9
        )

    def test_caps_inferred_batch_size_at_num_samples(self):
        segment = Segment(sequence="ATCG", sequence_type="dna")
        gen = _make_noop_generator(segment)
        gen.batch_size = 100

        assert (
            RejectionSamplingOptimizer._resolve_proposal_batch_size(
                generators=[gen],
                constraints=[],
                num_samples=12,
                configured=None,
            )
            == 12
        )


class TestRejectionSamplingOptimizerStandardMode:
    """Test RejectionSamplingOptimizer in standard mode (no energy_threshold)."""

    def test_rejection_sampling_optimizer_initialization(self):
        """Test basic RejectionSamplingOptimizer initialization in standard mode."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=10, num_results=5, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 10
        assert optimizer.proposal_batch_size == 1  # Default inferred from components
        assert optimizer.energy_threshold is None  # Standard mode
        assert optimizer.num_results == 5
        assert len(optimizer.constraints) == 1
        assert len(optimizer.generators) == 1

    def test_rejection_sampling_returns_k_constructs(self):
        """Test that Rejection Sampling optimizer returns exactly k constructs."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=20, num_results=3, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.result_sequences) == 3
        assert len(optimizer.energy_scores) == 3
        assert optimizer.num_results == 3

        # Verify energies are sorted (best first)
        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_rejection_sampling_keeps_best_proposals(self):
        """Test that Rejection Sampling keeps the best (lowest energy) proposals."""
        segment = Segment(sequence="AAAAAAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=2))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 80.0, "max_gc": 100.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=50, num_results=5, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        best_energy = optimizer.energy_scores[0]
        worst_energy = optimizer.energy_scores[-1]
        assert best_energy <= worst_energy

    def test_rejection_sampling_with_multiple_generators(self):
        """Test Rejection Sampling with multiple generators applied sequentially."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment)

        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=10, num_results=3, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.result_sequences) == 3
        assert len(optimizer.energy_scores) == 3

    def test_rejection_sampling_rounds_start_from_initial_state(self):
        """Test that each round starts from the initial state, not cumulative."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        initial_seq = segment.result_sequences[0].sequence

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=5, num_results=5, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        for i in range(5):
            seq = segment.result_sequences[i].sequence
            diff_count = sum(1 for a, b in zip(initial_seq, seq, strict=False) if a != b)
            # At most 1 mutation per round (may be 0 if random char matches original)
            assert diff_count <= 1, f"Expected <=1 mutation, got {diff_count} differences"

    def test_run_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=5, num_results=3, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Capture original state (base class initializes result_sequences to num_results by cycling)
        original_seq = segment.result_sequences[0].sequence
        assert original_seq == "ATCGATCG"
        assert len(segment.result_sequences) == 3  # Cycled from single source

        # First run
        optimizer.run()
        assert len(segment.result_sequences) == 3
        assert optimizer._initial_state is not None

        # Verify captured state contains cycled original sequences
        assert len(optimizer._initial_state["segments"]) == 1
        captured_result = optimizer._initial_state["segments"][0]["result"]
        assert len(captured_result) == 3  # Cycled to num_results
        assert all(s["sequence"] == original_seq for s in captured_result)

        # Verify energy scores captured
        assert "energy_scores" in optimizer._initial_state

        # Verify sorted list was populated (optimizer-specific state)
        assert len(optimizer._result_energies) == 3  # Has k entries after run

        # Manually modify sequences to invalid values to verify restore
        for seq in segment.result_sequences:
            seq.sequence = "GGGGGGGG"

        # Second run should restart - sorted list should be cleared and sequences restored
        optimizer.run()
        assert len(segment.result_sequences) == 3
        assert len(optimizer._result_energies) == 3  # Rebuilt from scratch

        # Verify sequences were restored (not all G's - restoration happened)
        assert any(seq.sequence != "GGGGGGGG" for seq in segment.result_sequences)

    def test_rejection_sampling_with_proposal_batch_size(self):
        """Test Rejection Sampling with proposal_batch_size > 1 for efficient batching."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=20, num_results=3, proposal_batch_size=5, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 20
        assert optimizer.proposal_batch_size == 5
        assert optimizer.num_results == 3

        optimizer.run()

        assert len(segment.result_sequences) == 3
        assert len(optimizer.energy_scores) == 3
        assert [entry["time_step"] for entry in optimizer.history] == list(range(1, 21))
        assert all(entry["optimizer"]["iteration_kind"] == "proposal" for entry in optimizer.history)

        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_samples_per_round_is_not_a_valid_config_field(self):
        """samples_per_round was removed; use proposal_batch_size for internal batching."""
        with pytest.raises(ValueError, match="samples_per_round"):
            RejectionSamplingOptimizerConfig(num_samples=10, num_results=5, samples_per_round=3)

    def test_proposal_batch_size_does_not_round_num_samples(self):
        """proposal_batch_size controls internal batches without changing num_samples."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # 10 samples with proposal_batch_size=3 -> batches of 3,3,3,1.
        config = RejectionSamplingOptimizerConfig(
            num_samples=10,
            num_results=5,
            proposal_batch_size=3,
            verbose=False,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 10
        assert optimizer.proposal_batch_size == 3

        optimizer.run()

        assert len(segment.result_sequences) == 5
        assert len(optimizer.energy_scores) == 5
        assert [entry["time_step"] for entry in optimizer.history] == list(range(1, 11))

    def test_inf_and_nan_energy_rejection(self):
        """Test that Rejection Sampling optimizer skips inf/nan energies."""
        import math

        from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
            GCContentConfig,
        )

        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            threshold=0.0,
        )

        config = RejectionSamplingOptimizerConfig(num_samples=100, num_results=5, verbose=False)

        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(optimizer.energy_scores) > 0
        assert len(optimizer.energy_scores) <= config.num_results
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy)
            assert not math.isnan(energy)


class TestRejectionSamplingOptimizerThresholdMode:
    """Test RejectionSamplingOptimizer in threshold mode (energy_threshold set)."""

    def test_threshold_mode_initialization(self):
        """Test RejectionSamplingOptimizer initialization in threshold mode."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=100, energy_threshold=0.5, num_results=3, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 100
        assert optimizer.energy_threshold == 0.5  # Threshold mode
        assert optimizer.num_results == 3

    def test_threshold_mode_stops_when_threshold_met(self):
        """Test that threshold mode stops early when threshold is met."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # High threshold that should be easily met
        config = RejectionSamplingOptimizerConfig(
            num_samples=1000,
            energy_threshold=100.0,  # Very high threshold, easily met
            num_results=3,
            verbose=False,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.result_sequences) == 3
        assert len(optimizer.energy_scores) == 3

    def test_threshold_mode_respects_num_samples(self):
        """Test that threshold mode stops at num_samples if threshold not met."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # Very low threshold that won't be met
        config = RejectionSamplingOptimizerConfig(
            num_samples=20,
            energy_threshold=0.0,  # Impossible to meet (energy would need to be negative)
            num_results=3,
            verbose=False,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Should have generated all num_samples and kept top k
        assert len(segment.result_sequences) == 3
        assert len(optimizer.energy_scores) == 3


class TestRejectionSamplingOptimizerValidation:
    """Test RejectionSamplingOptimizer config validation."""

    def test_num_results_cannot_exceed_num_samples(self):
        """Test that num_results cannot exceed num_samples."""
        with pytest.raises(ValueError, match="num_results \\(100\\) cannot exceed num_samples \\(10\\)"):
            _ = RejectionSamplingOptimizerConfig(
                num_samples=10,
                num_results=100,
            )

    def test_default_is_standard_mode(self):
        """Test that default (no energy_threshold) is standard mode."""
        config = RejectionSamplingOptimizerConfig(num_samples=10, num_results=5)
        assert config.energy_threshold is None


class TestRejectionSamplingOptimizerInternals:
    """Test RejectionSamplingOptimizer internal methods."""

    def test_result_sequences_always_sorted(self):
        """Test that result_sequences are always sorted by energy (ascending)."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=2))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=30, num_results=5)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # energy_scores should be sorted ascending
        assert optimizer.energy_scores == sorted(optimizer.energy_scores)
        # _result_energies should match energy_scores
        assert optimizer._result_energies == optimizer.energy_scores

    def test_empty_result_when_all_rejected(self):
        """Test that Rejection Sampling returns empty lists when all proposals are rejected."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=5, num_results=3)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Capture initial state and clear result
        optimizer._capture_initial_state()

        # Verify empty state
        assert optimizer._result_energies == []
        assert optimizer.energy_scores == []
        assert segment.result_sequences == []

    def test_all_proposals_rejected_by_filter(self):
        """Test Rejection Sampling optimizer handles case where all proposals are rejected by filter.

        This is a regression test for a bug where the optimizer would crash with
        RuntimeError when all proposals had inf/nan energies.
        """
        from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
            GCContentConfig,
        )

        segment = Segment(sequence="AAAAAAAAAA", sequence_type="dna")  # 0% GC
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(
                masking_strategy=MaskingStrategy(num_mutations=1)
            )  # Only 1 mutation, unlikely to reach 99% GC
        )
        gen.assign(segment)

        # Extremely strict filter - requires 99-100% GC content (effectively impossible)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=99.0, max_gc=100.0),
            threshold=0.0,  # Filter mode - rejected proposals get inf energy
        )

        config = RejectionSamplingOptimizerConfig(num_samples=20, num_results=5, verbose=False)

        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Should not crash - returns empty results when no valid proposals found
        optimizer.run()

        # No valid proposals found, empty results (no padding)
        assert len(optimizer.energy_scores) == 0
        assert len(segment.result_sequences) == 0
        assert optimizer.history
        assert all(entry["optimizer"]["filter_status"] == "failed" for entry in optimizer.history)
        assert all(entry["optimizer"]["failed_filter"] == constraint.label for entry in optimizer.history)

    def test_partial_proposals_rejected_by_filter(self):
        """Test Rejection Sampling optimizer handles case where some but not all proposals pass filter."""
        from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
            GCContentConfig,
        )

        segment = Segment(sequence="ATCGATCGAT", sequence_type="dna")  # 40% GC
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=2))
        )
        gen.assign(segment)

        # Moderate filter - requires 30-70% GC (some will pass, some won't)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=30.0, max_gc=70.0),
            threshold=0.0,  # Filter mode
        )

        config = RejectionSamplingOptimizerConfig(num_samples=50, num_results=10, verbose=False)

        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Should have results (up to k, may be fewer if some were rejected)
        assert len(optimizer.energy_scores) <= 10
        assert len(segment.result_sequences) == len(optimizer.energy_scores)


class TestRejectionSamplingOptimizerTrajectoryPreservation:
    """Test that Rejection Sampling preserves trajectory diversity from handoff."""

    def test_rejection_sampling_preserves_input_diversity(self):
        """Test that Rejection Sampling uses each proposal's own initial sequence, not just the first.

        This verifies the fix for the single-seed bug where Rejection Sampling was discarding
        diversity by always using proposals[0] as the mutation seed.
        """
        # Create segment with 3 distinct initial sequences (simulating handoff from previous optimizer)
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        # Pre-populate result_sequences with diverse seeds (simulating previous optimizer output)
        segment.result_sequences = [
            Sequence("AAAA", "dna"),
            Sequence("CCCC", "dna"),
            Sequence("GGGG", "dna"),
        ]

        gen = _make_noop_generator(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        # num_results=6 with 3 source sequences → cycling produces [A, C, G, A, C, G]
        # proposal_batch_size=6 means 6 proposals per internal batch
        config = RejectionSamplingOptimizerConfig(
            num_samples=6,  # Generate 6 samples total
            num_results=6,  # Keep top 6
            proposal_batch_size=6,
            verbose=False,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Initialize pools (this cycles through the 3 seeds to fill 6 slots)
        optimizer._initialize_sequence_pools()
        optimizer._capture_initial_state()

        # Run one proposal batch (no-op generator keeps sequences as their seeds)
        optimizer._run_proposal_batch(batch_num=1, first_proposal_number=1, batch_size=optimizer.proposal_batch_size)

        # Verify that proposals come from different seeds (cycled pattern)
        # With the fix: proposals should be [AAAA, CCCC, GGGG, AAAA, CCCC, GGGG]
        # With the bug: proposals would all be [AAAA, AAAA, AAAA, AAAA, AAAA, AAAA]
        proposals = [seq.sequence for seq in segment.proposal_sequences]

        # At least 2 unique sequences should be present (proving diversity is preserved)
        unique_proposals = set(proposals)
        assert len(unique_proposals) >= 2, (
            f"Rejection Sampling should preserve input diversity but found only {unique_proposals}. "
            f"This suggests all proposals are seeded from the first sequence."
        )

        # Verify the expected cycling pattern
        assert proposals == ["AAAA", "CCCC", "GGGG", "AAAA", "CCCC", "GGGG"], (
            f"Expected cycled pattern but got {proposals}"
        )

    def test_rejection_sampling_result_coherence_across_segments(self):
        """Test that result coherence is maintained across multiple segments.

        Each result index should use the same source index across all segments,
        preserving the semantic pairing from the previous optimizer.
        """
        # Create two segments with matching diverse seeds
        segment1 = Segment(sequence="AAAA", sequence_type="dna", label="seg1")
        segment2 = Segment(sequence="TTTT", sequence_type="dna", label="seg2")
        construct = Construct([segment1, segment2])

        # Pre-populate with paired sequences (index 0 pairs: AAAA-TTTT, index 1 pairs: CCCC-GGGG)
        segment1.result_sequences = [
            Sequence("AAAA", "dna"),
            Sequence("CCCC", "dna"),
        ]
        segment2.result_sequences = [
            Sequence("TTTT", "dna"),
            Sequence("GGGG", "dna"),
        ]

        gen1 = _make_noop_generator(segment1)
        gen2 = _make_noop_generator(segment2)

        # Use separate constraints for each segment
        constraint1 = Constraint(
            inputs=[segment1],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )
        constraint2 = Constraint(
            inputs=[segment2],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=4, num_results=4, proposal_batch_size=4, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint1, constraint2],
            config=config,
        )

        optimizer._initialize_sequence_pools()
        optimizer._capture_initial_state()
        optimizer._run_proposal_batch(batch_num=1, first_proposal_number=1, batch_size=optimizer.proposal_batch_size)

        # Verify result coherence: index i in segment1 should pair with index i in segment2
        proposals1 = [seq.sequence for seq in segment1.proposal_sequences]
        proposals2 = [seq.sequence for seq in segment2.proposal_sequences]

        # Expected: [AAAA, CCCC, AAAA, CCCC] and [TTTT, GGGG, TTTT, GGGG]
        assert proposals1 == ["AAAA", "CCCC", "AAAA", "CCCC"]
        assert proposals2 == ["TTTT", "GGGG", "TTTT", "GGGG"]

        # Verify pairing is preserved (same index = same source trajectory)
        for i in range(4):
            # Index 0,2 should both be from source 0 (AAAA-TTTT pair)
            # Index 1,3 should both be from source 1 (CCCC-GGGG pair)
            expected_source = i % 2
            if expected_source == 0:
                assert proposals1[i] == "AAAA" and proposals2[i] == "TTTT"
            else:
                assert proposals1[i] == "CCCC" and proposals2[i] == "GGGG"


class TestRejectionSamplingCustomLogging:
    """Regression: custom_logging must not corrupt results (Bug 1).

    Previously, logging could corrupt the sorted results list by reordering
    ``result_sequences`` while indices were still in use.
    """

    def test_custom_logging_does_not_corrupt_results(self):
        """Results with custom_logging must match results without it (same seed)."""
        seed = 42

        def run_rejection_sampling(custom_logging_fn=None):
            random.seed(seed)
            np.random.seed(seed)
            segment = Segment(sequence="ATCGATCG", sequence_type="dna")
            construct = Construct([segment])
            gen = RandomNucleotideGenerator(
                RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=2))
            )
            gen.assign(segment)
            constraint = Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config={"min_gc": 40.0, "max_gc": 60.0},
            )
            config = RejectionSamplingOptimizerConfig(
                num_samples=30,
                num_results=5,
                verbose=False,
                seed=seed,
            )
            optimizer = RejectionSamplingOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=config,
                custom_logging=custom_logging_fn,
            )
            optimizer.run()
            return (
                [s.sequence for s in segment.result_sequences],
                optimizer.energy_scores[:],
            )

        seqs_no_log, energies_no_log = run_rejection_sampling(custom_logging_fn=None)

        log_calls = []
        seqs_with_log, energies_with_log = run_rejection_sampling(custom_logging_fn=lambda r, s: log_calls.append(r))

        assert sorted(seqs_no_log) == sorted(seqs_with_log)
        assert sorted(energies_no_log) == sorted(energies_with_log)
        assert len(log_calls) > 0

    def test_custom_logging_callback_receives_segments(self):
        """Verify the custom_logging callback receives the correct arguments."""
        received = []

        def logger_fn(step, segments):
            received.append((step, len(segments)))

        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )
        config = RejectionSamplingOptimizerConfig(
            num_samples=5,
            num_results=3,
            verbose=False,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
            custom_logging=logger_fn,
        )
        optimizer.run()

        assert len(received) == 5
        for step, num_segments in received:
            assert isinstance(step, int)
            assert num_segments == 1


class TestRejectionSamplingLabelDeduplication:
    """Regression: optimizer must deduplicate constraint labels (Bug 3)."""

    def test_duplicate_constraint_labels_are_deduplicated(self):
        """Two constraints with the same label should be auto-renamed."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 20.0, "max_gc": 80.0},
        )
        assert constraint1.label == constraint2.label

        config = RejectionSamplingOptimizerConfig(num_samples=5, num_results=3, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint1, constraint2],
            config=config,
        )
        optimizer.run()
        assert constraint1.label != constraint2.label

    def test_deduplication_is_idempotent(self):
        """Calling _deduplicate_constraint_labels twice must not accumulate suffixes."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 20.0, "max_gc": 80.0},
        )

        config = RejectionSamplingOptimizerConfig(num_samples=5, num_results=3, verbose=False)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint1, constraint2],
            config=config,
        )

        optimizer._deduplicate_constraint_labels()
        label_after_first = constraint2.label
        optimizer._deduplicate_constraint_labels()
        label_after_second = constraint2.label

        assert label_after_first == label_after_second
        assert label_after_first.count("_1") == 1


class TestRejectionSamplingProposalTracking:
    """Test proposal_results tracking in Rejection Sampling history."""

    def test_proposal_tracking(self):
        """History has proposal_results for proposals that did not enter the top-k."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(
                num_samples=20,
                num_results=3,
                verbose=False,
                track_proposals=True,
            ),
        )
        optimizer.run()

        valid_rejectors = {DID_NOT_ENTER_TOP_K}
        all_rejectors = set()
        for entry in optimizer.history:
            metadata = entry["optimizer"]
            assert metadata["filter_status"] == "passed"
            assert metadata["failed_filter"] is None
            assert "accepted" not in metadata
            assert "rejected_by" not in metadata
            if "proposal_results" not in entry:
                continue
            for cand in entry["proposal_results"]:
                assert isinstance(cand["accepted"], bool)
                if cand["accepted"]:
                    assert cand["rejected_by"] is None
                else:
                    all_rejectors.add(cand["rejected_by"])

        assert all_rejectors.issubset(valid_rejectors)

    def test_proposal_rows_are_emitted_without_proposal_tracking_payload(self):
        """track_proposals=False still records proposal iterations, but omits proposal_results."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(
                num_samples=5,
                num_results=2,
                verbose=False,
                track_proposals=False,
            ),
        )
        optimizer.run()

        assert [entry["time_step"] for entry in optimizer.history] == [1, 2, 3, 4, 5]
        assert all(entry["optimizer"]["iteration_kind"] == "proposal" for entry in optimizer.history)
        assert all("proposal_results" not in entry for entry in optimizer.history)

    def test_final_proposal_is_not_duplicated_when_history_is_cleared_after_append(self):
        """API progress callbacks clear history after persisting each appended snapshot."""

        class ClearingHistory(list):
            def __init__(self):
                super().__init__()
                self.saved: list[dict] = []

            def append(self, entry):
                self.saved.append(entry)
                super().append(entry)
                self.clear()

        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(
                num_samples=5,
                num_results=2,
                proposal_batch_size=2,
                verbose=False,
            ),
        )
        history = ClearingHistory()
        optimizer.history = history

        optimizer.run()

        assert [entry["time_step"] for entry in history.saved] == [1, 2, 3, 4, 5]


class TestRejectionSamplingTrackingInterval:
    """Test tracking_interval in Rejection Sampling optimizer."""

    def test_tracking_interval(self):
        """tracking_interval=2 reduces history snapshots."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        config = RejectionSamplingOptimizerConfig(
            num_samples=10,
            num_results=3,
            verbose=False,
            tracking_interval=2,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        optimizer.run()

        # 10 proposal iterations with interval=2: proposals 2,4,6,8,10.
        saved_steps = {entry["time_step"] for entry in optimizer.history}
        assert saved_steps == {2, 4, 6, 8, 10}

    def test_tracking_interval_with_threshold_early_exit(self):
        """Threshold early-exit forces a final snapshot even on non-interval rounds."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        config = RejectionSamplingOptimizerConfig(
            num_samples=1000,
            num_results=3,
            verbose=False,
            tracking_interval=5,
            energy_threshold=100.0,  # Very high, easily met early
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        optimizer.run()

        # Threshold should be met well before proposal 1000
        assert len(optimizer.history) < config.num_samples

        # The last snapshot should reflect the round where threshold was met
        saved_steps = sorted(entry["time_step"] for entry in optimizer.history)
        last_saved = saved_steps[-1]
        # Final snapshot must exist and be > 0 (not just the initial snapshot)
        assert last_saved > 0
        # If threshold was met on a non-interval round, we still get a snapshot
        # (the fix ensures this)


class TestRejectionSamplingMetadata:
    """Test metadata preservation through Rejection Sampling optimization."""

    def test_rejection_sampling_preserves_initial_metadata_on_result(self):
        """Initial user metadata should survive through Rejection Sampling rounds to result_sequences."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna", metadata={"user_key": "user_value"})
        construct = Construct([segment])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )

        config = RejectionSamplingOptimizerConfig(
            num_samples=5,
            num_results=3,
            verbose=False,
        )
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        for seq in segment.result_sequences:
            assert seq._metadata.get("user_key") == "user_value"
