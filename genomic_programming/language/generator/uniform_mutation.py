"""
UniformMutationGenerator for random point mutations.

A sequence generator that proposes random point mutations.
"""

from typing import Callable, Optional, final
import random

from ..base import Generator, Segment


@final
class UniformMutationGenerator(Generator):
    """
    A sequence generator that proposes random point mutations.

    This generator initializes with a random sequence and samples single-nucleotide
    or amino acid mutations on each call to sample().

    Examples:
        Creating a DNA mutation generator:
        >>> segment = Segment(sequence="ATCGG", sequence_type=SequenceType.DNA)
        >>> gen = UniformMutationGenerator(
        ...     batch_size=5,
        ...     sequence_length=5,
        ...     num_mutations=2
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces 2 random mutations
        >>> outputs = gen.get_generator_outputs()
        >>> len(outputs[0])  # 5 (batch size)

        Note: Using a mutation scheduler. Define a function that takes the iteration count 
            and returns the number of mutations.
        
        >>> def mutation_scheduler(iteration: int) -> int:
        ...     return max(1, 10 - iteration // 10)  # Decrease mutations over time
        >>> gen = UniformMutationGenerator(
        ...     batch_size=5,
        ...     sequence_length=100,
        ...     mutation_scheduler=mutation_scheduler
        ... )
    """

    def __init__(
        self,
        batch_size: int = 1,
        sequence_length: int = 100,
        num_mutations: int = 1,
        mutation_scheduler: Optional[Callable[[int], int]] = None,
    ) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            batch_size: Number of sequence variants to maintain simultaneously.
            sequence_length: Length of the sequence to generate.
            num_mutations: Number of mutations to introduce per sequence per sample() call.
                          Ignored if mutation_scheduler is provided.
            mutation_scheduler: Optional callable that takes iteration count and returns
                              number of mutations. If provided, overrides num_mutations.
        """
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length
        self.num_mutations = num_mutations
        self.mutation_scheduler = mutation_scheduler

    def assign(
        self, assigned_segments: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.
        """
        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True

        initial_sequence = self._generator_output.batch_sequences[0].sequence
        valid_chars = self._generator_output._valid_chars - set(" ")
        valid_chars_list = list(valid_chars)
        if initial_sequence == "":
            self._generator_output.batch_sequences[0].sequence = "".join(
                random.choice(valid_chars_list) for _ in range(self.sequence_length)
            )
        else:
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )
        self._generator_output.create_batch(self.batch_size)

        # No model initialization needed for this generator
        self._is_initialized = True

    def sample(self) -> None:
        """
        Introduce random point mutations in each sequence.

        For each sequence in the batch, selects random positions and replaces
        the characters with different random characters from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Determine number of mutations for this iteration
        if self.mutation_scheduler is not None:
            current_mutations = self.mutation_scheduler(self.iteration_count)
        else:
            current_mutations = self.num_mutations

        # Sample mutations for each output in the segment batch
        for sequence in self._generator_output.batch_sequences:
            current_sequence = sequence.sequence
            sequence_length = len(current_sequence)
            
            # Ensure we don't try to mutate more positions than available
            actual_mutations = min(current_mutations, sequence_length)
            
            # Select random positions to mutate (without replacement)
            positions_to_mutate = random.sample(range(sequence_length), actual_mutations)
            
            # Apply mutations
            for pos in positions_to_mutate:
                current_char = current_sequence[pos]
                
                # Make sure the mutated character is different from the current one
                possible_mutations = [
                    c for c in self._generator_output._valid_chars if c != current_char
                ]
                mutated_char = random.choice(possible_mutations)
                current_sequence = (
                    current_sequence[:pos]
                    + mutated_char
                    + current_sequence[pos + 1 :]
                )
            
            sequence.sequence = current_sequence

        # Increment iteration count (shared helper on base class)
        self._increment_iteration_count()

