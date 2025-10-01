"""
SlowMutationGenerator for testing and demonstration.

A generator that introduces mutations slowly with configurable delays.
"""

from typing import final
import random
import time

from ..base import Generator, Segment


@final
class SlowMutationGenerator(Generator):
    """A generator that introduces mutations slowly for testing and demonstration purposes."""
    
    def __init__(self, batch_size: int = 1, sequence_length: int = 20, sleep_time: float = 2.0):
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length
        self.sleep_time = sleep_time
        
    def assign(self, assigned_segments: Segment) -> None:
        """Assign a Segment to this generator.
        
        Args:
            assigned_segments: A single Segment to be assigned to this generator.
        """
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        
        # Initialize with random sequence if empty
        if self._generator_output.batch_sequences[0].sequence == "":
            valid_chars = list(self._generator_output._valid_chars - set(" "))
            initial_sequence = "".join(
                random.choice(valid_chars) for _ in range(self.sequence_length)
            )
            self._generator_output.batch_sequences[0].sequence = initial_sequence
            
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True
        
    def sample(self):
        """Sample with a sleep to simulate slow processing."""
        self._validate_generator()
        
        time.sleep(self.sleep_time)
        
        # Simple mutation logic
        for sequence in self._generator_output.batch_sequences:
            if len(sequence.sequence) > 0:
                # Mutate a random position
                mutated_index = random.randint(0, len(sequence.sequence) - 1)
                current_sequence = sequence.sequence
                current_char = current_sequence[mutated_index]
                
                # Get valid mutations
                possible_mutations = [
                    c for c in self._generator_output._valid_chars 
                    if c != current_char and c != " "
                ]
                
                if possible_mutations:
                    mutated_char = random.choice(possible_mutations)
                    new_sequence = (
                        current_sequence[:mutated_index] + 
                        mutated_char + 
                        current_sequence[mutated_index + 1:]
                    )
                    sequence.sequence = new_sequence

