from __future__ import annotations
import pytest
import math
from unittest.mock import MagicMock, patch
from typing import List

from proto_language.language.core import (
    Optimizer,
    Construct,
    Segment,
    Constraint,
    Generator,
    Sequence,
)

# Concrete implementation for testing the abstract base class
class ConcreteOptimizer(Optimizer):
    """Concrete implementation of Optimizer for testing purposes."""
    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        num_candidates: int,
        num_selected: int,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
        verbose: bool = False,
    ) -> None:
        """
        Implementation of abstract __init__.
        Passes arguments directly to super().
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_candidates=num_candidates,
            num_selected=num_selected,
            clear_tool_cache=clear_tool_cache,
            verbose=verbose
        )

    def run(self) -> None:
        """Dummy run implementation."""
        pass


class MockGenerator(Generator):
    """Minimal mock generator."""
    def __init__(self):
        super().__init__()
        self._assigned_segment = None

    def assign(self, segment: Segment) -> None:
        self._assigned_segment = segment
        self._assigned_segment._is_assigned = True

    def sample(self, *args, **kwargs) -> None:
        pass


def _setup_base_optimizer_components(
    num_candidates: int = 4,
    num_selected: int = 2,
):
    """Helper function to set up components for testing Optimizer."""
    # 1. Create segment
    segment = Segment(sequence="ATCG", sequence_type="dna")

    # 2. Create construct
    construct = Construct([segment])

    # 3. Create generator
    generator = MockGenerator()
    generator.assign(segment)

    # 4. Create dummy constraint
    constraint = MagicMock(spec=Constraint)
    constraint.inputs = [segment]
    constraint.label = "MockConstraint"
    constraint.threshold = None  # Default to scoring constraint
    constraint.weight = 1.0

    # Mock evaluate to return a list of 1.0s by default
    # Note: evaluate returns a list of floats (scores) or bools (if used as filter)
    constraint.evaluate.return_value = [1.0] * num_candidates

    return construct, generator, constraint, segment


class TestOptimizer:
    """Tests for the base Optimizer class functionality."""

    def test_initialization_validation(self):
        """Tests that optimizer validates inputs correctly."""
        construct, generator, constraint, segment = _setup_base_optimizer_components()

        # Test 1: Valid initialization
        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=4,
            num_selected=2,
        )
        assert optimizer.num_candidates == 4
        assert optimizer.num_selected == 2

        # Test 2: Empty lists
        with pytest.raises(ValueError, match="Constructs list cannot be empty"):
            ConcreteOptimizer([], [generator], [constraint], 4, 2)

        with pytest.raises(ValueError, match="Generators list cannot be empty"):
            ConcreteOptimizer([construct], [], [constraint], 4, 2)

        with pytest.raises(ValueError, match="Constraints list cannot be empty"):
            ConcreteOptimizer([construct], [generator], [], 4, 2)

        # Test 3: Unassigned generator
        unassigned_gen = MockGenerator()
        with pytest.raises(RuntimeError, match="has no segment assigned"):
            ConcreteOptimizer([construct], [unassigned_gen], [constraint], 4, 2)

        # Test 4: Constraint with no inputs
        empty_constraint = MagicMock(spec=Constraint)
        empty_constraint.inputs = []
        with pytest.raises(RuntimeError, match="has no input segment"):
            ConcreteOptimizer([construct], [generator], [empty_constraint], 4, 2)

    def test_sequence_pool_initialization_fresh(self):
        """Tests initialization of sequence pools when no previous candidates exist."""
        construct, generator, constraint, segment = _setup_base_optimizer_components(
            num_candidates=5, num_selected=2
        )
        # Ensure segment starts fresh
        segment.candidate_sequences = []
        segment.selected_sequences = []
        original_seq = "ATCG"
        segment.original_sequence = Sequence(sequence=original_seq, sequence_type="dna")

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=5,
            num_selected=2,
        )
        assert optimizer.constructs == [construct] # Needed for linter.

        # 1. Selected sequences should be initialized from original (copies)
        assert len(segment.selected_sequences) == 2
        assert all(s.sequence == original_seq for s in segment.selected_sequences)

        # 2. Candidate sequences should be initialized from original
        assert len(segment.candidate_sequences) == 5
        assert all(s.sequence == original_seq for s in segment.candidate_sequences)

        # 3. Verify independence (deep copy)
        segment.selected_sequences[0].sequence = "GGGG"
        assert segment.candidate_sequences[0].sequence == original_seq

    def test_sequence_pool_initialization_chained(self):
        """Tests initialization when inheriting sequences from previous optimizer."""
        construct, generator, constraint, segment = _setup_base_optimizer_components(
            num_candidates=5, num_selected=3
        )

        # Simulate results from a previous run
        prev_best = Sequence(sequence="AAAA", sequence_type="dna")
        prev_second = Sequence(sequence="TTTT", sequence_type="dna")
        segment.selected_sequences = [prev_best, prev_second]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=5,
            num_selected=3, # Requesting 3, but only 2 available
        )
        assert optimizer.constructs == [construct] # Needed for linter.

        # 1. Should use existing selected sequences
        assert segment.selected_sequences[0].sequence == "AAAA"
        assert segment.selected_sequences[1].sequence == "TTTT"

        # 2. Should pad with copies of the best (first) sequence if not enough
        assert len(segment.selected_sequences) == 3
        assert segment.selected_sequences[2].sequence == "AAAA"

        # 3. Candidates should be initialized from the best sequence
        assert len(segment.candidate_sequences) == 5
        assert all(s.sequence == "AAAA" for s in segment.candidate_sequences)

    def test_score_energy_add(self):
        """Tests basic energy scoring with 'add' operation."""
        construct, generator, constraint1, segment = _setup_base_optimizer_components(
            num_candidates=2
        )

        # Setup Constraint 1: weight 1.0, scores [10.0, 20.0]
        constraint1.weight = 1.0
        constraint1.threshold = None
        constraint1.evaluate.return_value = [10.0, 20.0]

        # Setup Constraint 2: weight 0.5, scores [4.0, 8.0]
        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.weight = 1.0
        constraint2.threshold = None
        constraint2.evaluate.return_value = [4.0, 8.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint1, constraint2],
            num_candidates=2,
            num_selected=2,
        )

        # Ensure segment has candidates so scoring loop runs
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]

        optimizer.score_energy(operation="add")

        # Expected:
        # Cand 0: 1.0*10.0 + 1.0*4.0 = 14.0
        # Cand 1: 1.0*20.0 + 1.0*8.0 = 28.0
        assert optimizer.energy_scores == [14.0, 28.0]

    def test_score_energy_multiply(self):
        """Tests energy scoring with 'multiply' operation."""
        construct, generator, constraint1, segment = _setup_base_optimizer_components(
            num_candidates=2
        )

        # Setup Constraint 1: weight 1.0, scores [2.0, 3.0]
        constraint1.weight = 1.0
        constraint1.evaluate.return_value = [2.0, 3.0]

        # Setup Constraint 2: weight 2.0, scores [4.0, 5.0]
        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.weight = 1.0
        constraint2.threshold = None
        constraint2.evaluate.return_value = [4.0, 5.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint1, constraint2],
            num_candidates=2,
            num_selected=2,
        )
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]

        optimizer.score_energy(operation="multiply")

        # Expected:
        # Cand 0: (1.0*2.0) * (1.0*4.0) = 8.0
        # Cand 1: (1.0*3.0) * (1.0*5.0) = 15.0
        assert optimizer.energy_scores == [8.0, 15.0]

    def test_score_energy_filter_rejection(self):
        """Tests that filter constraints correctly reject candidates."""
        construct, generator, constraint1, segment = _setup_base_optimizer_components(
            num_candidates=3
        )
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # Constraint 1: Filter. Rejects index 1.
        constraint1.threshold = 0.5
        # evaluate returns dense booleans: [Pass, Fail, Pass]
        constraint1.evaluate.return_value = [True, False, True]

        # Constraint 2: Scoring.
        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.threshold = None
        constraint2.weight = 1.0
        # Returns dense scores (NaN for skipped candidate 1)
        constraint2.evaluate.return_value = [10.0, float('nan'), 20.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint1, constraint2],
            num_candidates=3,
            num_selected=2,
        )

        optimizer.score_energy(operation="add", filter_penalty=999.0)

        # Expected:
        # Cand 0: Passed -> 10.0
        # Cand 1: Rejected -> 999.0 (penalty overrides 0.0)
        # Cand 2: Passed -> 20.0
        assert optimizer.energy_scores == [10.0, 999.0, 20.0]

    def test_score_energy_sparse_evaluation(self):
        """Tests that subsequent constraints are NOT evaluated for rejected candidates."""
        construct, generator, constraint1, segment = _setup_base_optimizer_components(
            num_candidates=3
        )
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # Constraint 1: Filter. Rejects index 1.
        constraint1.threshold = 0.0
        constraint1.evaluate.return_value = [True, False, True]

        # Constraint 2: Scoring. Returns dense results (NaN for skipped candidate 1)
        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.threshold = None
        constraint2.evaluate.return_value = [1.0, float('nan'), 1.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint1, constraint2],
            num_candidates=3,
            num_selected=2,
        )

        optimizer.score_energy()

        # Check that Constraint 2 was called with a mask that reflects Constraint 1's rejection
        # Constraint 1 rejected candidate 1, so mask should be [True, False, True]
        _, kwargs = constraint2.evaluate.call_args
        assert kwargs['mask'] == [True, False, True]

    def test_score_energy_filter_penalty(self):
        """
        Tests that a rejected sequence receives infinite energy.

        With the preprocessing approach, filters are evaluated first, then scoring
        constraints are only evaluated on passing candidates.
        """
        construct, generator, scoring_constraint, segment = _setup_base_optimizer_components(
            num_candidates=2
        )
        segment.candidate_sequences = [
            Sequence("PASS", "protein"),
            Sequence("FAIL", "protein"),
        ]

        # Constraint 1: Scoring constraint (only evaluates passing candidate)
        scoring_constraint.threshold = None
        scoring_constraint.weight = 1.0
        # With preprocessing: filter runs first, only candidate 0 passes
        # Scoring returns dense [10.0, NaN] (NaN for unevaluated candidate 1)
        scoring_constraint.evaluate.return_value = [10.0, float('nan')]

        # Constraint 2: Filter constraint (Rejects second candidate)
        filter_constraint = MagicMock(spec=Constraint)
        filter_constraint.inputs = [segment]
        filter_constraint.threshold = 0.5
        filter_constraint.weight = 1.0
        # Filter evaluates all candidates, returns dense [True, False]
        filter_constraint.evaluate.return_value = [True, False]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[scoring_constraint, filter_constraint],
            num_candidates=2,
            num_selected=2,
        )

        # Run scoring with infinite penalty
        optimizer.score_energy(operation="add", filter_penalty=float('inf'))

        # Candidate 0: passes filter, scores 10.0
        # Candidate 1: fails filter, receives filter_penalty (inf)
        assert optimizer.energy_scores[0] == 10.0
        assert math.isinf(optimizer.energy_scores[1])

    def test_tool_cache_clearing(self):
        """Tests that tool cache clearing logic works as expected."""
        construct, generator, constraint, segment = _setup_base_optimizer_components()

        # Mock the tool cache
        with patch('proto_language.language.core.optimizer.ToolCache') as MockCache:
            mock_cache_instance = MockCache.return_value
            mock_cache_instance.current_size = 200

            # Case 1: clear_tool_cache = int (threshold)
            # Threshold (100) < Current (200) -> Should prune
            optimizer = ConcreteOptimizer(
                [construct], [generator], [constraint], 4, 2,
                clear_tool_cache=100
            )
            optimizer._clear_tool_cache()
            mock_cache_instance.prune.assert_called_with(100)

            # Reset
            mock_cache_instance.prune.reset_mock()

            # Case 2: clear_tool_cache = True -> Clear all
            optimizer.clear_tool_cache = True
            optimizer._clear_tool_cache()
            mock_cache_instance.clear.assert_called_once()

            # Case 3: clear_tool_cache = List[str] -> Clear specific
            mock_cache_instance.clear.reset_mock()
            optimizer.clear_tool_cache = ["tool_A"]
            optimizer._clear_tool_cache()
            mock_cache_instance.clear.assert_called_with("tool_A")

    def test_progress_snapshot(self):
        """Tests that history snapshots capture correct state."""
        construct, generator, constraint, segment = _setup_base_optimizer_components(
            num_candidates=2, num_selected=1
        )
        optimizer = ConcreteOptimizer(
            [construct], [generator], [constraint], 2, 1
        )

        optimizer.energy_scores = [0.1, 0.9]
        optimizer._save_progress_snapshot(time_step=5)

        assert len(optimizer.history) == 1
        snapshot = optimizer.history[0]

        assert snapshot["time_step"] == 5
        # Should only save top 'num_selected' scores
        assert snapshot["energy_scores"] == [0.1]
        assert "constructs" in snapshot
        # Ensure deep copy
        assert snapshot["constructs"] is not optimizer.constructs

    def test_score_energy_invalid_operation(self):
        """Tests that invalid operations raise ValueError."""
        construct, generator, constraint, segment = _setup_base_optimizer_components()
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]

        with pytest.raises(ValueError, match="Operation must be 'add' or 'multiply'"):
            optimizer.score_energy(operation="invalid")

    def test_score_energy_inconsistent_state_raises_error(self):
        """Tests that RuntimeError is raised when a passed candidate has NaN score.

        This tests the inconsistent state check: if a candidate passes all filters,
        it should have been evaluated by all scoring constraints and should not have NaN.
        """
        construct, generator, constraint, segment = _setup_base_optimizer_components(
            num_candidates=3
        )
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # No filter constraints - all candidates should pass
        constraint.threshold = None
        constraint.weight = 1.0
        # Bug simulation: constraint returns NaN for candidate 1 even though it should be evaluated
        constraint.evaluate.return_value = [1.0, float('nan'), 1.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=3,
            num_selected=2,
        )

        # Should raise RuntimeError because candidate 1 passed all filters but has NaN score
        with pytest.raises(RuntimeError, match="Inconsistent state: candidate 1 passed all filters but has NaN score"):
            optimizer.score_energy(operation="add")
