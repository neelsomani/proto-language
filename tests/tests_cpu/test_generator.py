import pytest
import random
from typing import Tuple

import sys

sys.path.append(".")
from proto_language.base import (
    Construct,
    ConstructSegment,
    Constraint,
    SequenceType,
)
from proto_language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.generator import UniformMutationGenerator, TwoSegmentUniformMutationGenerator, MCMCGenerator, ChainedGenerator

# Helper function
def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.DNA
) -> ConstructSegment:
    """Helper to create a ConstructSegment with a single sequence."""
    return ConstructSegment(sequence=sequence, sequence_type=seq_type)


class TestUniformMutationGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        gen = UniformMutationGenerator(sequence_length=15, batch_size=5)
        assert gen.sequence_length == 15
        assert gen.batch_size == 5
        assert not gen._is_initialized

    def test_assign_and_initialization(self):
        """Tests the assign method initializes the output segment correctly."""
        seq_len = 20
        # Test assign with an empty segment (should initialize randomly)
        segment = create_segment("", seq_type=SequenceType.RNA)
        gen = UniformMutationGenerator(sequence_length=seq_len, batch_size=3)
        gen.assign(segment)

        assert gen._is_initialized
        assert gen._generator_output is segment
        assert segment._is_assigned
        assert len(segment) == 3
        assert len(segment[0]) == seq_len
        assert all(c in "ACGU" for c in segment[0].sequence)

        # Test assign with a pre-defined sequence
        predefined_seq = "A" * seq_len
        segment_pre = create_segment(predefined_seq, seq_type=SequenceType.RNA)
        gen.assign(segment_pre)
        assert segment_pre[0].sequence == predefined_seq

    def test_assign_errors(self):
        """Tests runtime validation for the assign method."""
        gen = UniformMutationGenerator(sequence_length=10)
        # Should raise error if provided sequence length doesn't match configured length
        with pytest.raises(AssertionError):
            gen.assign(create_segment("A"*5))

    def test_sample_mutates_sequence(self):
        """Tests the sample method introduces a single valid mutation."""
        seq_len = 25
        gen = UniformMutationGenerator(sequence_length=seq_len)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.PROTEIN)
        gen.assign(segment)

        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence

        assert len(mutated_sequence) == seq_len
        # Check that exactly one position has changed
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 1
        diff_indices = [i for i, (a, b) in enumerate(zip(initial_sequence, mutated_sequence)) if a != b]
        mutated_char = mutated_sequence[diff_indices[0]]

        assert mutated_char in segment._valid_chars
        assert mutated_char != initial_sequence[diff_indices[0]]

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch independently."""
        gen = UniformMutationGenerator(sequence_length=30, batch_size=5)
        segment = create_segment("A" * 30)
        gen.assign(segment)

        initial_sequences = [s.sequence for s in segment]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment]

        for i in range(len(initial_sequences)):
            assert initial_sequences[i] != mutated_sequences[i]
            diff_count = sum(1 for a,b in zip(initial_sequences[i], mutated_sequences[i]) if a != b)
            assert diff_count == 1
        # Check that mutations are likely different across the batch
        assert len(set(mutated_sequences)) > 1

    def test_deterministic_behavior(self):
        """Tests that with a fixed seed, the behavior is reproducible."""
        def run_with_seed(seed):
            random.seed(seed)
            gen = UniformMutationGenerator(sequence_length=50)
            segment = create_segment("", seq_type=SequenceType.DNA)
            gen.assign(segment)
            initial_seq = segment[0].sequence
            for _ in range(10):
                gen.sample()
            final_seq = segment[0].sequence
            return initial_seq, final_seq

        init1, final1 = run_with_seed(42)
        init2, final2 = run_with_seed(42)
        init3, final3 = run_with_seed(123)

        assert init1 == init2
        assert final1 == final2
        assert init1 != init3
        assert final1 != final3

    def test_sample_len_one_sequence(self):
        """Tests that a sequence of length 1 is mutated correctly."""
        gen = UniformMutationGenerator(sequence_length=1)
        segment = create_segment("A", seq_type=SequenceType.DNA)
        gen.assign(segment)

        initial_char = segment[0].sequence
        gen.sample()
        mutated_char = segment[0].sequence

        assert len(mutated_char) == 1
        assert mutated_char in "CGT"
        assert mutated_char != initial_char

    def test_num_mutations_parameter(self):
        """Tests that specifying num_mutations produces exactly that many changes."""
        seq_len = 30
        num_mut = 5
        gen = UniformMutationGenerator(sequence_length=seq_len, num_mutations=num_mut)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        gen.assign(segment)

        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence

        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == num_mut

    def test_num_mutations_capped_by_sequence_length(self):
        """Tests that num_mutations larger than length is capped to sequence length."""
        seq_len = 3
        num_mut = 10
        gen = UniformMutationGenerator(sequence_length=seq_len, num_mutations=num_mut)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        gen.assign(segment)

        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence

        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == seq_len

    def test_mutation_scheduler_decreasing(self):
        """Tests that a scheduler can control mutations based on iteration count."""
        seq_len = 20
        def scheduler(iteration: int) -> int:
            # 1st call: 3, 2nd: 2, 3rd+: 1
            return max(1, 3 - iteration)

        gen = UniformMutationGenerator(sequence_length=seq_len, mutation_scheduler=scheduler)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        gen.assign(segment)

        # Iteration 0 -> expect 3 mutations
        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 3
        assert gen.get_iteration_count() == 1

        # Iteration 1 -> expect 2 mutations
        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 2
        assert gen.get_iteration_count() == 2

        # Iteration 2 -> expect 1 mutation
        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 1
        assert gen.get_iteration_count() == 3

    def test_iteration_count_independent_instances(self):
        """Tests iteration counters are per generator instance and resettable."""
        seq_len = 10
        g1 = UniformMutationGenerator(sequence_length=seq_len)
        g2 = UniformMutationGenerator(sequence_length=seq_len)
        s1 = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        s2 = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        g1.assign(s1)
        g2.assign(s2)

        assert g1.get_iteration_count() == 0
        assert g2.get_iteration_count() == 0

        g1.sample()
        assert g1.get_iteration_count() == 1
        assert g2.get_iteration_count() == 0

        g2.sample()
        g2.sample()
        assert g1.get_iteration_count() == 1
        assert g2.get_iteration_count() == 2

        g1.reset_iteration_count()
        assert g1.get_iteration_count() == 0

class TestTwoSegmentUniformMutationGenerator:
    def test_assign_and_sample(self):
        """Tests basic functionality: assign two segments and mutate them."""
        segment1 = create_segment("ATCGG", seq_type=SequenceType.DNA)
        segment2 = create_segment("MKLLF", seq_type=SequenceType.PROTEIN)
        
        gen = TwoSegmentUniformMutationGenerator(batch_size=1)
        gen.assign([segment1, segment2])

        assert gen._is_initialized
        assert len(gen.get_generator_outputs()) == 2
        
        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence
        
        gen.sample()
        
        # Both sequences should be mutated
        assert segment1[0].sequence != initial_seq1
        assert segment2[0].sequence != initial_seq2
        # Lengths should be preserved
        assert len(segment1[0].sequence) == len(initial_seq1)
        assert len(segment2[0].sequence) == len(initial_seq2)

    def test_assign_errors(self):
        """Tests error conditions for assignment."""
        gen = TwoSegmentUniformMutationGenerator()
        
        # Wrong number of segments
        with pytest.raises(ValueError, match="requires exactly 2 segments"):
            gen.assign([create_segment("ATCG")])
        
        # Empty sequences
        with pytest.raises(ValueError, match="requires segments with existing sequences"):
            gen.assign([create_segment(""), create_segment("ATCG")])

    def test_different_lengths(self):
        """Tests that segments can have different lengths."""
        segment1 = create_segment("AT")
        segment2 = create_segment("GCGCGCGC")
        
        gen = TwoSegmentUniformMutationGenerator()
        gen.assign([segment1, segment2])
        
        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence
        
        gen.sample()
        
        assert len(segment1[0].sequence) == 2
        assert len(segment2[0].sequence) == 8
        assert segment1[0].sequence != initial_seq1
        assert segment2[0].sequence != initial_seq2

def _setup_mcmc_components(
    seq_length: int = 10,
    batch_size: int = 1,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC generator for testing."""
    # 1. Create the proposal generator and the segment it will modify.
    proposal_gen = UniformMutationGenerator(
        sequence_length=seq_length, batch_size=batch_size
    )
    segment = create_segment("A" * seq_length) # Start with a known sequence
    proposal_gen.assign(segment)

    # 2. Create the construct and constraint.
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        },
    )

    # 3. Create the MCMC generator.
    mcmc_gen = MCMCGenerator(
        constructs=[construct],
        generators=[proposal_gen],
        constraints=[constraint],
        num_steps=num_mcmc_steps,
        verbose=False,
    )
    return mcmc_gen, proposal_gen, constraint, segment


class TestMCMCGenerator:
    def test_initialization_and_validation(self):
        """Tests successful initialization and validation of MCMCGenerator."""
        mcmc_gen, proposal_gen, constraint, segment = _setup_mcmc_components()
        
        assert mcmc_gen.generators == [proposal_gen]
        assert mcmc_gen.constraints == [constraint]
        assert mcmc_gen.constraint_weights == [1.0]
        assert mcmc_gen._is_initialized # IterativeGenerator base class is auto-initialized

        # Test validation errors
        # Unassigned generator
        unassigned_gen = UniformMutationGenerator(sequence_length=10)
        with pytest.raises(RuntimeError, match="has not been assigned"):
            MCMCGenerator(
                constructs=[Construct([create_segment("A"*10)])],
                generators=[unassigned_gen],
                constraints=[],
            )
        
        # Mismatched weights and constraints
        with pytest.raises(ValueError, match="must match"):
            MCMCGenerator(
                constructs=mcmc_gen.constructs,
                generators=mcmc_gen.generators,
                constraints=mcmc_gen.constraints,
                constraint_weights=[1.0, 2.0],
            )

        # Unassigned segment in construct
        segment_assigned = create_segment("A"*10)
        gen = UniformMutationGenerator(sequence_length=10)
        gen.assign(segment_assigned)
        segment_unassigned = create_segment("C"*10) # Not assigned to any generator
        construct = Construct([segment_assigned, segment_unassigned])
        # Need at least one constraint, so add a dummy one
        dummy_constraint = Constraint(
            inputs=[segment_assigned],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        with pytest.raises(ValueError, match="not assigned to any generator"):
            MCMCGenerator(
                constructs=[construct],
                generators=[gen],
                constraints=[dummy_constraint]
            )

    def test_score_energy(self):
        """Tests the score_energy method."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))

        # Test with a sequence that is within the target GC range
        segment.batch_sequences[0].sequence = "GCGCGAATTA"  # 50% GC
        mcmc_gen.score_energy()
        assert len(mcmc_gen.energy_scores) == 1
        assert mcmc_gen.energy_scores[0] == 0.0

        # Test with a sequence below the target range
        segment.batch_sequences[0].sequence = "GCTTAATTAA"  # 20% GC
        mcmc_gen.score_energy()
        expected_score = (40.0 - 20.0) / 40.0  # 0.5
        assert abs(mcmc_gen.energy_scores[0] - expected_score) < 1e-9

        # Test that energy scores are stored in the generator's energy_scores attribute
        assert hasattr(mcmc_gen, 'energy_scores')
        assert len(mcmc_gen.energy_scores) == 1
        assert abs(mcmc_gen.energy_scores[0] - expected_score) < 1e-9
        
        # Test that calling score_energy again updates the stored scores
        segment.batch_sequences[0].sequence = "GCGCGCGCGC"  # 100% GC -> score 1.0
        mcmc_gen.score_energy()
        expected_new_score = abs((40.0 - 100.0) / 40.0)  # Should be 1.5, but clamped to 1.0
        assert abs(mcmc_gen.energy_scores[0] - min(expected_new_score, 1.0)) < 1e-9

    def test_score_energy_multiply(self):
        """Tests the score_energy method with operation='multiply'."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))
        segment.batch_sequences[0].sequence = "GCTTAATTAA"  # 20% GC -> score 0.5
        
        # With one constraint, multiply and add should be the same
        mcmc_gen.score_energy(operation="add")
        energy_add = mcmc_gen.energy_scores[0]
        mcmc_gen.score_energy(operation="multiply")
        energy_mul = mcmc_gen.energy_scores[0]
        assert abs(energy_add - 0.5) < 1e-9
        assert abs(energy_mul - 0.5) < 1e-9

    def test_sample_history(self):
        """Tests that sampling can improve the energy score over time."""
        # Use a restrictive constraint to guide optimization
        mcmc_gen, _, _, segment = _setup_mcmc_components(
            seq_length=50,
            gc_target_range=(80.0, 90.0), # Encourage high GC
            num_mcmc_steps=100
        )
        
        # Start with a bad sequence
        segment.batch_sequences[0].sequence = "A" * 50
        mcmc_gen.score_energy()
        initial_energy = mcmc_gen.energy_scores[0]
        assert initial_energy > 0.99 # Should be max penalty (1.0)
        
        # Sample and check for improvement
        mcmc_gen.sample()
        mcmc_gen.score_energy()
        final_energy = mcmc_gen.energy_scores[0]
        
        assert final_energy < initial_energy
        assert len(mcmc_gen.history) > 1 # Check that history is being tracked

    def test_multiple_constraints(self):
        """Tests the MCMC generator with multiple constraints and weights."""
        seq_len = 30
        proposal_gen = UniformMutationGenerator(sequence_length=seq_len)
        segment = create_segment("A" * seq_len)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        gc_con = Constraint(
            [segment], gc_content_constraint, {"min_gc": 40.0, "max_gc": 60.0}
        )
        len_con = Constraint(
            [segment], sequence_length_constraint, {"target_length": seq_len}
        )

        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_con, len_con],
            constraint_weights=[1.0, 2.0], # Weight length more
            num_steps=1,
            verbose=False,
        )

        segment.batch_sequences[0].sequence = "A" * 20 # Violates length and GC
        gc_score = gc_con.evaluate()[0] # (40-0)/40 = 1.0
        len_score = len_con.evaluate()[0] # (30-20)/30 = 0.333
        
        # E = 1.0 * 1.0 + 2.0 * 0.333...
        expected_energy = gc_score * 1.0 + len_score * 2.0
        mcmc_gen.score_energy("add")
        assert abs(mcmc_gen.energy_scores[0] - expected_energy) < 1e-9

        # Test multiply operation
        expected_energy_mul = (gc_score * 1.0) * (len_score * 2.0)
        mcmc_gen.score_energy("multiply")
        assert abs(mcmc_gen.energy_scores[0] - expected_energy_mul) < 1e-9

    def test_with_multiple_generators(self):
        """Tests MCMC with more than one proposal generator."""
        # Create a second, simple generator for testing
        class InversionGenerator(UniformMutationGenerator):
            def sample(self) -> None:
                for seq in self._generator_output.batch_sequences:
                    # Invert a small slice of the sequence
                    start = random.randint(0, len(seq.sequence) - 3)
                    end = start + 3
                    sub_seq = seq.sequence[start:end]
                    inverted_sub = sub_seq[::-1]
                    seq.sequence = seq.sequence[:start] + inverted_sub + seq.sequence[end:]
        
        seq_len = 50
        # Generator 1: Point mutations
        mut_gen = UniformMutationGenerator(sequence_length=seq_len)
        segment1 = create_segment("A" * seq_len)
        mut_gen.assign(segment1)

        # Generator 2: Inversions
        inv_gen = InversionGenerator(sequence_length=seq_len)
        segment2 = create_segment("C" * seq_len)
        inv_gen.assign(segment2)

        construct = Construct([segment1, segment2])
        constraint = Constraint(
            inputs=[segment1, segment2], # Constraint on the whole construct
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": seq_len * 2}
        )

        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[mut_gen, inv_gen],
            constraints=[constraint],
            num_steps=20,
            verbose=False,
        )

        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence
        
        # Sampling should modify the sequences
        mcmc_gen.sample()

        final_seq1 = segment1[0].sequence
        final_seq2 = segment2[0].sequence

        # Check that at least one sequence was modified (both should be, but inversions might be symmetric)
        assert initial_seq1 != final_seq1 or initial_seq2 != final_seq2


def _setup_chained_components(
    seq_length: int = 10,
    batch_size: int = 2,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
):
    """Helper function to set up components for ChainedGenerator testing."""
    # 1. Create segments and generators
    segment1 = create_segment("A" * seq_length)
    segment2 = create_segment("C" * seq_length)
    
    gen1 = UniformMutationGenerator(sequence_length=seq_length, batch_size=batch_size)
    gen2 = UniformMutationGenerator(sequence_length=seq_length, batch_size=batch_size)
    
    # 2. Assign generators to segments (this sets _is_assigned = True)
    gen1.assign(segment1)
    gen2.assign(segment2)
    
    # 3. Create constructs and constraints
    construct1 = Construct([segment1])
    construct2 = Construct([segment2])
    
    constraint1 = Constraint(
        inputs=[segment1],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        }
    )
    constraint2 = Constraint(
        inputs=[segment2],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        }
    )
    
    return segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2


class TestChainedGenerator:
    def test_initialization(self):
        """Tests successful initialization of ChainedGenerator."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=3,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        assert len(chained.generator_stages) == 2
        assert chained.generator_stages[0] == stage1
        assert chained.generator_stages[1] == stage2
        assert chained.verbose == False
        assert chained.capture_metadata == True
        assert len(chained.stage_results) == 0
        assert chained._execution_start_time is None

    def test_validation_errors(self):
        """Tests validation errors during initialization."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Test empty stages list
        with pytest.raises(ValueError, match="At least one generator stage must be provided"):
            ChainedGenerator([], verbose=False)
        
        # Test non-IterativeGenerator stage
        with pytest.raises(ValueError, match="must be an IterativeGenerator"):
            ChainedGenerator([gen1], verbose=False)  # gen1 is not an IterativeGenerator
        
        # Test mismatched batch sizes
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        
        # Create stage with different batch size
        gen2_different_batch = UniformMutationGenerator(sequence_length=10, batch_size=3)
        gen2_different_batch.assign(segment2)
        stage2_different = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2_different_batch],
            constraints=[constraint2],
            num_steps=3,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="same batch_size"):
            ChainedGenerator([stage1, stage2_different], verbose=False)


    def test_basic_execution(self):
        """Tests basic execution of the chained generator."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Run the pipeline
        chained.run()
        
        # Check that results were captured
        assert len(chained.stage_results) == 2
        assert chained.stage_results[0]['stage'] == 0
        assert chained.stage_results[1]['stage'] == 1
        assert chained.stage_results[0]['stage_type'] == 'MCMCGenerator'
        assert chained.stage_results[1]['stage_type'] == 'MCMCGenerator'

    def test_sequence_propagation(self):
        """Tests that sequences are properly propagated between stages."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Run the pipeline
        chained.run()
        
        # Check that sequences were propagated
        # Stage 1 should have modified segment1
        stage1_constructs = chained.stage_results[0]['constructs']
        stage2_constructs = chained.stage_results[1]['constructs']
        
        # The sequences should be different from the initial "A" * 10
        # Access the sequence through the batch_sequences
        assert stage1_constructs[0].segments[0].batch_sequences[0].sequence != "A" * 10
        assert stage2_constructs[0].segments[0].batch_sequences[0].sequence != "C" * 10

    def test_metadata_capture(self):
        """Tests that metadata is properly captured from each stage."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=True, capture_metadata=True)
        
        # Run the pipeline
        chained.run()
        
        # Check metadata capture
        for i, result in enumerate(chained.stage_results):
            assert 'stage' in result
            assert 'stage_type' in result
            assert 'constructs' in result
            assert 'final_energy' in result
            assert 'execution_time' in result
            assert 'stage_config' in result
            assert 'outputs_metadata' in result
            
            # Check specific values
            assert result['stage'] == i
            assert result['execution_time'] > 0
            assert len(result['constructs']) > 0

    def test_results_access_methods(self):
        """Tests all the results access methods."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test before running
        with pytest.raises(RuntimeError, match="run\\(\\) must be called"):
            chained.get_final_constructs()
        
        # Run the pipeline
        chained.run()
        
        # Test get_final_constructs
        final_constructs = chained.get_final_constructs()
        assert len(final_constructs) > 0
        assert final_constructs == chained.stage_results[-1]['constructs']
        
        # Test get_final_sequences
        final_sequences = chained.get_final_sequences()
        assert len(final_sequences) > 0
        assert isinstance(final_sequences[0], str)
        
        # Test get_stage_results
        stage_results = chained.get_stage_results()
        assert len(stage_results) == 2
        assert stage_results == chained.stage_results
        
        # Test get_stage_metadata
        stage_metadata = chained.get_stage_metadata()
        assert len(stage_metadata) == 2
        for meta in stage_metadata:
            assert 'stage' in meta
            assert 'stage_type' in meta
            assert 'outputs_metadata' in meta
            assert 'execution_summary' in meta

    def test_stage_access_methods(self):
        """Tests methods for accessing individual stages."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test get_stage
        assert chained.get_stage(0) == stage1
        assert chained.get_stage(1) == stage2
        assert chained.get_stage(2) is None
        assert chained.get_stage(-1) is None
        
        # Test get_stage_result before running
        assert chained.get_stage_result(0) is None
        
        # Run the pipeline
        chained.run()
        
        # Test get_stage_result after running
        stage1_result = chained.get_stage_result(0)
        stage2_result = chained.get_stage_result(1)
        assert stage1_result is not None
        assert stage2_result is not None
        assert stage1_result['stage'] == 0
        assert stage2_result['stage'] == 1
        assert chained.get_stage_result(2) is None

    def test_execution_summary(self):
        """Tests the execution summary functionality."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test summary before running
        summary_before = chained.get_execution_summary()
        assert summary_before['total_stages'] == 2
        assert summary_before['total_execution_time'] == 0.0
        assert summary_before['final_energy'] is None
        assert summary_before['energy_progression'] == []
        assert summary_before['stage_types'] == ['MCMCGenerator', 'MCMCGenerator']
        
        # Run the pipeline
        chained.run()
        
        # Test summary after running
        summary_after = chained.get_execution_summary()
        assert summary_after['total_stages'] == 2
        assert summary_after['total_execution_time'] > 0
        assert summary_after['final_energy'] is not None
        assert len(summary_after['energy_progression']) == 2
        assert len(summary_after['stage_types']) == 2

    def test_energy_progression(self):
        """Tests the energy progression tracking."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test before running
        assert chained.get_energy_progression() == []
        
        # Run the pipeline
        chained.run()
        
        # Test after running
        energy_prog = chained.get_energy_progression()
        assert len(energy_prog) == 2
        assert all(isinstance(e, (float, type(None))) for e in energy_prog)

    def test_export_results(self):
        """Tests the export functionality."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test export before running
        with pytest.raises(RuntimeError, match="No results to export"):
            chained.export_results('test.json')
        
        # Run the pipeline
        chained.run()
        
        # Test JSON export
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            chained.export_results(temp_file, 'json')
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Test pickle export
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            chained.export_results(temp_file, 'pickle')
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            chained.export_results('test.txt', 'txt')

    def test_verbose_execution(self):
        """Tests that verbose mode provides appropriate output."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=True)
        
        # Run the pipeline (this should print progress)
        chained.run()
        
        # Check that results were captured despite verbose output
        assert len(chained.stage_results) == 2

    def test_metadata_capture_disabled(self):
        """Tests that metadata capture can be disabled."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False, capture_metadata=False)
        
        # Run the pipeline
        chained.run()
        
        # Check that basic results are still captured
        assert len(chained.stage_results) == 2
        
        # Check that outputs_metadata might be empty or minimal
        for result in chained.stage_results:
            assert 'outputs_metadata' in result

    def test_single_stage_execution(self):
        """Tests execution with only one stage."""
        segment1, _, gen1, _, construct1, _, constraint1, _ = _setup_chained_components()
        
        # Create single stage
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1], verbose=False)
        
        # Run the pipeline
        chained.run()
        
        # Check results
        assert len(chained.stage_results) == 1
        assert chained.stage_results[0]['stage'] == 0
        assert chained.stage_results[0]['stage_type'] == 'MCMCGenerator'
        
        # Test final constructs access
        final_constructs = chained.get_final_constructs()
        assert len(final_constructs) > 0
