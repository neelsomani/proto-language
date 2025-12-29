"""
Tests for TopKOptimizer functionality.

Minimal tests verifying core behavior of the TopKOptimizer.
"""

import pytest
from proto_language.language.core import (
    Construct, Segment, Constraint)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig
)
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig
from proto_language.language.constraint import gc_content_constraint, sequence_length_constraint


class TestTopKOptimizer:
    """Test core TopKOptimizer functionality."""

    def test_topk_optimizer_initialization(self):
        """Test basic TopKOptimizer initialization."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = TopKOptimizerConfig(min_num_samples=10, k=5, batch_size=1, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.min_num_samples == 10
        assert optimizer.batch_size == 1
        assert optimizer.rounds == 10
        assert optimizer.k == 5
        assert optimizer.num_selected == 5
        assert len(optimizer.constraints) == 1
        assert len(optimizer.generators) == 1

    def test_topk_returns_k_constructs(self):
        """Test that TopK optimizer returns exactly k constructs."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=1
            )
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(min_num_samples=20, k=3, batch_size=1, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3
        assert optimizer.num_selected == 3

        # Verify energies are sorted (best first)
        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_topk_keeps_best_candidates(self):
        """Test that TopK keeps the best (lowest energy) candidates."""
        segment = Segment(sequence="AAAAAAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=2
            )
        )
        gen.assign(segment)

        # Constraint that prefers higher GC content (80-100%)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 80.0, "max_gc": 100.0},
        )

        config = TopKOptimizerConfig(min_num_samples=50, k=5, batch_size=1, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Best candidate should have lower energy than worst
        best_energy = optimizer.energy_scores[0]
        worst_energy = optimizer.energy_scores[-1]

        assert best_energy <= worst_energy

    def test_topk_k_capped_at_rounds(self):
        """Test that k cannot exceed number of candidates."""
        # Validation now happens at config creation time
        with pytest.raises(ValueError, match="k \\(100\\) cannot exceed min_num_samples \\(10\\)"):
            _ = TopKOptimizerConfig(min_num_samples=10, k=100, batch_size=1, verbose=False)

    def test_topk_with_multiple_generators(self):
        """Test TopK with multiple generators applied sequentially."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=1
            )
        )
        gen1.assign(segment)

        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=1
            )
        )
        gen2.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(min_num_samples=10, k=3, batch_size=1, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

    def test_topk_rounds_start_from_initial_state(self):
        """Test that each round starts from the initial state, not cumulative."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        initial_seq = segment.selected_sequences[0].sequence

        # Generator that mutates 1 position
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=1
            )
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )

        config = TopKOptimizerConfig(min_num_samples=5, k=5, batch_size=1, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # All top-k sequences should differ from initial by only 1 mutation
        # (since each round starts fresh and applies 1 mutation)
        for i in range(5):
            seq = segment.selected_sequences[i].sequence
            diff_count = sum(1 for a, b in zip(initial_seq, seq) if a != b)
            assert diff_count == 1, f"Expected 1 mutation, got {diff_count} differences"

    def test_topk_with_batch_size(self):
        """Test TopK with batch_size > 1 for efficient batching."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=1
            )
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # Generate 20 total candidates in batches of 5
        # Should result in 4 rounds × 5 candidates per round = 20 total
        config = TopKOptimizerConfig(min_num_samples=20, k=3, batch_size=5, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Verify derived parameters
        assert optimizer.min_num_samples == 20
        assert optimizer.batch_size == 5
        assert optimizer.rounds == 4  # 20 / 5 = 4 rounds
        assert optimizer.k == 3

        optimizer.run()

        # Should return exactly k=3 best candidates
        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

        # Verify energies are sorted (best first)
        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_topk_batch_size_validation(self):
        """Test that min_num_samples must be divisible by batch_size."""
        with pytest.raises(ValueError, match="min_num_samples \\(10\\) must be divisible by batch_size \\(3\\)"):
            _ = TopKOptimizerConfig(min_num_samples=10, k=5, batch_size=3, verbose=False)

    def test_topk_with_energy_threshold(self):
        """Test TopK with energy_threshold for adaptive stopping."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                num_mutations=1
            )
        )
        gen.assign(segment)

        # Constraint that prefers GC content (penalizes sequences with all As)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # Start with minimum 10 candidates, but continue until worst < 1.0 or hit 100 max
        config = TopKOptimizerConfig(
            min_num_samples=10,
            k=3,
            batch_size=2,
            energy_threshold=1.0,
            max_num_samples=100,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Verify config and derived parameters
        assert optimizer.energy_threshold == 1.0
        assert optimizer.max_num_samples == 100

        optimizer.run()

        # Should return exactly k=3 candidates
        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

        # Worst in top-k should be below threshold (or max candidates reached)
        worst_energy = optimizer.energy_scores[-1]
        assert worst_energy < 1.0 or optimizer.min_num_samples >= 100

    def test_topk_threshold_validation(self):
        """Test validation of threshold-related parameters."""
        # max_num_samples must be divisible by batch_size
        with pytest.raises(ValueError, match="max_num_samples \\(15\\) must be divisible by batch_size \\(4\\)"):
            _ = TopKOptimizerConfig(
                min_num_samples=8,
                k=3,
                batch_size=4,
                energy_threshold=0.5,
                max_num_samples=15,
                verbose=False
            )

        # max_num_samples must be >= min_num_samples
        with pytest.raises(ValueError, match="max_num_samples \\(50\\) must be >= min_num_samples \\(100\\)"):
            _ = TopKOptimizerConfig(
                min_num_samples=100,
                k=10,
                batch_size=10,
                energy_threshold=0.5,
                max_num_samples=50,
                verbose=False
            )

    def test_infinite_energy_rejection(self):
        """Test that TopK optimizer skips inf/nan energies from heap."""
        import math
        from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
        
        # Start with moderate GC sequence so mutations can go both above and below threshold
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")  # 50% GC
        construct = Construct([segment])
        
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=3)
        )
        gen.assign(segment)
        
        # Use GC content constraint with threshold=0 to convert scores to boolean
        # This will return True (inf energy) for sequences outside the range
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            threshold=0.0,  # Convert to boolean filter: True (inf) if outside range, False (0.0) if inside
        )
        
        config = TopKOptimizerConfig(
            min_num_samples=100,  # Generate many samples, some will have inf energy, some won't
            k=5,
            batch_size=10,
            verbose=False
        )
        
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        
        optimizer.run()
        
        # Verify all returned sequences have finite energies
        assert len(optimizer.energy_scores) > 0, "Should have at least some valid sequences"
        assert len(optimizer.energy_scores) <= config.k
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy), f"Found infinite energy {energy} in results - filtering failed!"
            assert not math.isnan(energy), f"Found NaN energy in results - filtering failed!"
        
        # Verify all selected sequences pass the threshold (GC 40-60%)
        for seq_obj in segment.selected_sequences:
            gc_count = sum(1 for nt in seq_obj.sequence.upper() if nt in 'GC')
            gc_percent = 100.0 * gc_count / len(seq_obj.sequence)
            assert 40.0 <= gc_percent <= 60.0, f"Sequence with GC {gc_percent}% should have been rejected (inf energy)"
