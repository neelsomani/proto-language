"""
TwoSegmentUniformMutationGenerator for paired sequence mutations.

A sequence generator that proposes random point mutations across two segments.
"""

from typing import Iterable, final
import random

from ..base import Generator, Segment


@final
class TwoSegmentUniformMutationGenerator(Generator):
    """
    A sequence generator that proposes random point mutations across two segments.

    This generator is specifically designed to work with exactly two Segment objects,
    randomly mutating each segment independently. This is a common pattern in bio models that model
    paired sequences (e.g., protein-ligand, protein-protein, or DNA-RNA pairs). The segments can have different lengths.

    Examples:
        Creating a two-segment mutation generator:
        >>> segment1 = Segment(sequence="ATCGG", sequence_type=SequenceType.DNA)
        >>> segment2 = Segment(sequence="GCTAA", sequence_type=SequenceType.DNA)
        >>> gen = TwoSegmentUniformMutationGenerator(batch_size=5)
        >>> gen.assign([segment1, segment2])
        >>> gen.sample()  # Introduces random mutations in each segment
        >>> outputs = gen.get_generator_outputs()
        >>> len(outputs)  # 2 (number of segments)
    """

    def __init__(
        self,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the two-segment mutation generator.

        Args:
            batch_size: Number of sequence variants to maintain simultaneously.
        """
        super().__init__(batch_size=batch_size)

    def assign(
        self, assigned_segments: Iterable[Segment]
    ) -> None:
        """
        Assign exactly two Segment objects to this generator.

        Args:
            assigned_segments: An iterable of exactly two Segment objects.

        Raises:
            ValueError: If not exactly two segments are provided.
        """
        segments = list(assigned_segments)
        
        if len(segments) != 2:
            raise ValueError(f"TwoSegmentUniformMutationGenerator requires exactly 2 segments, got {len(segments)}")

        # Segments can have different lengths (common in bio models like BindCraft and RF diffusion)
        # Initialize _generator_outputs for two segments
        self._generator_outputs = tuple(segments)
        for segment in self._generator_outputs:
            segment._is_assigned = True
            
            # Validate that existing sequences are not empty
            initial_sequence = segment.batch_sequences[0].sequence
            if initial_sequence == "":
                raise ValueError("TwoSegmentUniformMutationGenerator requires segments with existing sequences (cannot be empty)")
            
            segment.create_batch(self.batch_size)

        # No model initialization needed for this generator
        self._is_initialized = True

    def sample(self) -> None:
        """
        Introduce a random point mutation in each sequence of each segment.

        For each sequence in each segment's batch, selects a random position and replaces
        the character with a different random character from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Sample mutation for each segment
        for segment in self._generator_outputs:
            for sequence in segment.batch_sequences:
                if len(sequence.sequence) == 0:
                    continue  # Skip empty sequences
                    
                mutated_index = random.randint(0, len(sequence.sequence) - 1)
                current_sequence = sequence.sequence
                current_char = current_sequence[mutated_index]

                # Make sure the mutated character is different from the current one
                possible_mutations = [
                    c for c in segment._valid_chars if c != current_char
                ]
                if possible_mutations:
                    mutated_char = random.choice(possible_mutations)
                    sequence.sequence = (
                        current_sequence[:mutated_index]
                        + mutated_char
                        + current_sequence[mutated_index + 1 :]
                    )

