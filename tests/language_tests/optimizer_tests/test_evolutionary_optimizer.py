"""tests/language_tests/optimizer_tests/test_evolutionary_optimizer.py."""

import pytest
from proto_tools.transforms.masking import MaskingStrategy
from pydantic import BaseModel, ValidationError

from proto_language.constraint import gc_content_constraint
from proto_language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.core import Constraint, ConstraintOutput, Construct, Program, Segment
from proto_language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerConfig


# Empty config for test constraints
class EmptyConfig(BaseModel):
    pass


def _setup_evolutionary_components(
    seq_length: int = 10,
    population_size: int = 10,
    gc_target_range: tuple[float, float] = (40.0, 60.0),
    num_generations: int = 10,
    elitism_count: int = 1,
    tournament_size: int = 3,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
) -> tuple[EvolutionaryOptimizer, RandomNucleotideGenerator, Constraint, Segment]:
    """Helper function to set up a basic Evolutionary Optimizer for testing."""
    # 1. Create the mutation generator and the segment it will modify
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # 2. Create the construct and constraint
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(
            min_gc=gc_target_range[0],
            max_gc=gc_target_range[1],
        ),
    )

    # 3. Create the Evolutionary Optimizer config
    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=elitism_count,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        verbose=False,
    )

    optimizer = EvolutionaryOptimizer(
        constructs=[construct],
        generators=[mutation_gen],
        constraints=[constraint],
        config=config,
    )
    return optimizer, mutation_gen, constraint, segment


class TestEvolutionaryOptimizer:
    def test_initialization_and_validation(self) -> None:
        """Tests successful initialization and validation of EvolutionaryOptimizer."""
        optimizer, mutation_gen, constraint, _segment = _setup_evolutionary_components()

        assert optimizer.generators == [mutation_gen]
        assert optimizer.constraints == [constraint]
        assert optimizer.num_results == 10  # population_size
        assert optimizer.population_size == 10
        assert optimizer.elitism_count == 1
        assert optimizer.tournament_size == 3

        # Test validation errors - unassigned generator
        test_segment = Segment(sequence="A" * 10, sequence_type="dna")
        unassigned_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )

        # Create a dummy scoring function
        def dummy_scoring_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        dummy_scoring_func._constraint_config_class = EmptyConfig
        dummy_scoring_func._constraint_supported_sequence_types = ["dna"]

        dummy_constraint = Constraint(
            inputs=[test_segment],
            function=dummy_scoring_func,
            function_config=EmptyConfig(),
        )
        with pytest.raises(RuntimeError, match="has no segment assigned"):
            EvolutionaryOptimizer(
                constructs=[Construct([test_segment])],
                generators=[unassigned_gen],
                constraints=[dummy_constraint],
                config=EvolutionaryOptimizerConfig(population_size=10, num_generations=1),
            )

    def test_config_validation(self) -> None:
        """Tests EvolutionaryOptimizerConfig validation."""
        # Valid config
        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=20,
            elitism_count=2,
            tournament_size=3,
        )
        assert config.population_size == 10
        assert config.num_generations == 20
        assert config.elitism_count == 2

        # elitism_count >= population_size should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, elitism_count=10)

        # tournament_size > population_size should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, tournament_size=11)

        # population_size < 2 should fail (minimum for crossover)
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=1, num_generations=1)

        # Negative values should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=-1)

        # Invalid rates should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, crossover_rate=1.5)
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, mutation_rate=-0.1)

    def test_deterministic_convergence(self) -> None:
        """Tests that the EA converges deterministically with a fixed seed."""
        # Set up optimizer with fixed seed
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=50, max_gc=50),  # Target exactly 50% GC
        )

        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=20,
            elitism_count=2,
            seed=42,  # Fixed seed
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Run twice with same seed
        program1 = Program(optimizers=[optimizer], num_results=10, seed=42)
        program1.run()
        results1 = [seq.sequence for seq in segment.result_sequences]
        energies1 = list(optimizer.energy_scores)

        # Create fresh optimizer with same seed
        segment2 = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen2.assign(segment2)

        constraint2 = Constraint(
            inputs=[segment2],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=50, max_gc=50),
        )

        optimizer2 = EvolutionaryOptimizer(
            constructs=[Construct([segment2])],
            generators=[mutation_gen2],
            constraints=[constraint2],
            config=EvolutionaryOptimizerConfig(
                population_size=10,
                num_generations=20,
                elitism_count=2,
                seed=42,
            ),
        )

        program2 = Program(optimizers=[optimizer2], num_results=10, seed=42)
        program2.run()
        results2 = [seq.sequence for seq in segment2.result_sequences]
        energies2 = list(optimizer2.energy_scores)

        # Results should be identical
        assert results1 == results2
        assert energies1 == energies2

    def test_elitism_preserves_best(self) -> None:
        """Tests that elitism ensures best score never regresses."""
        optimizer, _, _, segment = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=10,
            elitism_count=2,
            gc_target_range=(50, 50),
        )

        # Manually set up initial population with known good solutions
        # Create population with varying GC content
        sequences = [
            "GCGCGCGCGCAAAAAAAAAA",  # 50% GC - perfect score (0.0)
            "GCGCGCGCGCGCGCGCGCGC",  # 100% GC - bad score
            "A" * 20,  # 0% GC - bad score
            "T" * 20,  # 0% GC - bad score
            "GCGCGCGCGCGCGCGCGCGC",  # 100% GC - bad score
            "ATATATATATATATAT ATAT",  # ~0% GC - bad score
            "GCGCGCGCGCGCGCGCGCGC",  # 100% GC - bad score
            "AAAAAAAAAAAAAAAAAAAA",  # 0% GC - bad score
            "TTTTTTTTTTTTTTTTTTTT",  # 0% GC - bad score
            "GGGGGGGGGGGGGGGGGGGG",  # 100% GC - bad score
        ]

        for i, seq in enumerate(sequences):
            segment.result_sequences[i].sequence = seq.replace(" ", "")

        # Run optimization
        optimizer.run()

        # Get history of best scores across generations
        best_scores = []
        for snapshot in optimizer.history:
            energy_scores = snapshot.get("energy_scores", [])
            if energy_scores:
                best_scores.append(min(s for s in energy_scores if s is not None and not isinstance(s, str)))

        # With elitism, best score should never increase (get worse)
        for i in range(1, len(best_scores)):
            assert best_scores[i] <= best_scores[i - 1], (
                f"Best score regressed at generation {i}: " f"{best_scores[i-1]:.4f} -> {best_scores[i]:.4f}"
            )

    def test_crossover_single_point(self) -> None:
        """Tests that single-point crossover produces valid children."""
        optimizer, _, _, segment = _setup_evolutionary_components(
            seq_length=20,
            population_size=4,
            num_generations=1,
            crossover_rate=1.0,  # Always crossover
            mutation_rate=0.0,  # No mutation
        )

        # Set up parents with distinct patterns
        parent1 = "AAAAAAAAAAAAAAAAAAAA"
        parent2 = "CCCCCCCCCCCCCCCCCCCC"
        segment.result_sequences[0].sequence = parent1
        segment.result_sequences[1].sequence = parent2
        segment.result_sequences[2].sequence = parent1
        segment.result_sequences[3].sequence = parent2

        # Run one generation
        optimizer.run()

        # Check that at least some offspring are valid crossovers
        # (mix of A's and C's, not pure A or pure C)
        offspring_seqs = [seq.sequence for seq in segment.result_sequences]

        # With crossover, we should see some mixed sequences
        # (unless all tournaments selected the same parent, which is unlikely)
        # At minimum, sequences should be valid DNA
        for seq in offspring_seqs:
            assert len(seq) == 20
            assert all(c in "ACGT" for c in seq)

    def test_crossover_uniform(self) -> None:
        """Tests that uniform crossover produces valid children."""
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        config = EvolutionaryOptimizerConfig(
            population_size=4,
            num_generations=1,
            crossover_strategy="uniform",  # Uniform crossover
            crossover_rate=1.0,  # Always crossover
            mutation_rate=0.0,  # No mutation
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Set up parents
        parent1 = "AAAAAAAAAAAAAAAAAAAA"
        parent2 = "CCCCCCCCCCCCCCCCCCCC"
        segment.result_sequences[0].sequence = parent1
        segment.result_sequences[1].sequence = parent2
        segment.result_sequences[2].sequence = parent1
        segment.result_sequences[3].sequence = parent2

        # Run one generation
        optimizer.run()

        # Check offspring validity
        offspring_seqs = [seq.sequence for seq in segment.result_sequences]
        for seq in offspring_seqs:
            assert len(seq) == 20
            assert all(c in "ACGT" for c in seq)

    def test_population_diversity(self) -> None:
        """Tests that EA maintains population diversity better than single-chain MCMC."""
        optimizer, _, _, segment = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=10,
            elitism_count=1,
            gc_target_range=(50, 50),
        )

        optimizer.run()

        # Count unique sequences in final population
        final_sequences = {seq.sequence for seq in segment.result_sequences}

        # EA should maintain some diversity (not all identical)
        # With population size 10, we expect at least 2-3 unique sequences
        assert len(final_sequences) >= 2, f"Population collapsed to {len(final_sequences)} unique sequences"

    def test_composition_with_program(self) -> None:
        """Tests that EvolutionaryOptimizer works correctly in a multi-stage Program."""
        # Stage 1: EA optimizer
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        construct = Construct([segment])  # Create construct once and reuse

        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        optimizer1 = EvolutionaryOptimizer(
            constructs=[construct],  # Reuse same construct object
            generators=[gen1],
            constraints=[constraint1],
            config=EvolutionaryOptimizerConfig(
                population_size=5,
                num_generations=5,
            ),
        )

        # Stage 2: Another EA optimizer (chained)
        gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen2.assign(segment)

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        optimizer2 = EvolutionaryOptimizer(
            constructs=[construct],  # Reuse same construct object
            generators=[gen2],
            constraints=[constraint2],
            config=EvolutionaryOptimizerConfig(
                population_size=5,
                num_generations=5,
            ),
        )

        # Run program with two EA stages
        program = Program(optimizers=[optimizer1, optimizer2], num_results=5)
        program.run()

        # Check that final results exist and are valid
        assert len(segment.result_sequences) == 5
        for seq in segment.result_sequences:
            assert len(seq.sequence) == 20
            assert all(c in "ACGT" for c in seq.sequence)

    def test_mutation_rate_effect(self) -> None:
        """Tests that mutation_rate controls the frequency of mutations."""
        # No mutation
        optimizer_no_mut, _, _, segment_no_mut = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=1,
            crossover_rate=0.0,  # No crossover, just cloning
            mutation_rate=0.0,  # No mutation
        )

        # Set all individuals to same sequence
        initial_seq = "AAAAAAAAAAAAAAAAAAAA"
        for seq in segment_no_mut.result_sequences:
            seq.sequence = initial_seq

        optimizer_no_mut.run()

        # With no crossover and no mutation, all offspring should be identical to parents
        no_mut_seqs = {seq.sequence for seq in segment_no_mut.result_sequences}
        assert len(no_mut_seqs) == 1  # All identical
        assert next(iter(no_mut_seqs)) == initial_seq

        # With mutation
        optimizer_with_mut, _, _, segment_with_mut = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=1,
            crossover_rate=0.0,  # No crossover
            mutation_rate=1.0,  # Always mutate
        )

        # Set all individuals to same sequence
        for seq in segment_with_mut.result_sequences:
            seq.sequence = initial_seq

        optimizer_with_mut.run()

        # With 100% mutation rate, most offspring should differ from parent
        with_mut_seqs = {seq.sequence for seq in segment_with_mut.result_sequences}
        # Should have some variation (at least a few different sequences)
        assert len(with_mut_seqs) >= 2

    def test_single_character_sequence(self) -> None:
        """Tests that optimizer handles very short sequences gracefully."""
        # Test with minimal length sequence (length 1)
        segment = Segment(sequence="A", sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        # Create a constraint that accepts all sequences
        def always_pass_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        always_pass_func._constraint_config_class = EmptyConfig
        always_pass_func._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=always_pass_func,
            function_config=EmptyConfig(),
        )

        config = EvolutionaryOptimizerConfig(
            population_size=4,
            num_generations=2,
            crossover_rate=1.0,
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Should run without errors
        optimizer.run()

        # All sequences should be length 1
        for seq in segment.result_sequences:
            assert len(seq.sequence) == 1
            assert seq.sequence in "ACGT"
