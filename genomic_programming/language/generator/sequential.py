"""
Sequential Generator

Extracted from generator.py for better code organization.
"""

from typing import Any, Callable, List, Optional, Tuple, Dict, final
import copy
import random
import sys

import numpy as np

from ..base import IterativeGenerator, Construct, Generator, Constraint, Sequence

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0


@final
class SequentialGenerator(IterativeGenerator):
    """
    Sequential generator for chaining autoregressive sequence generators.

    Applies multiple generators in sequence where each uses the previous generator's
    output as input prompts. After all generators run, accepts or rejects the
    combined changes based on energy improvement and temperature annealing.

    Requirements:
    - All generators must output exactly one Segment
    - Generators after the first must accept prompt_seqs parameter in sample()

    Examples:
        Basic sequential chaining:
        >>> constructs = [Construct([segment1, segment2])]
        >>> sequential = SequentialGenerator(
        ...     constructs=constructs,
        ...     generators=[gen1, gen2, gen3],  # Chain: gen1 -> gen2(gen1_out) -> gen3(gen2_out)
        ...     constraints=[constraint1, constraint2],
        ...     constraint_weights=[1.0, 2.0],  # Weight constraint2 more heavily
        ...     num_steps=50,
        ...     temperature=0.8,  # Accept/reject after all generators
        ...     temperature_min=0.001
        ... )
        >>> sequential.sample()
        >>> final_sequences = sequential.constructs

    Notes:
        - Final sequences: initial_prompt + gen1_output + gen2_output + ...
        - Temperature annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        num_steps: int = 1,
        temperature: float = 1.0,
        temperature_min: float = 0.0001,
        track_step_size: int = 1,
        custom_logging: Optional[Callable[[int, Sequence], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the sequential generator with ordered sub-generators.

        Args:
            constructs: List of Construct objects to be optimized.
            generators: List of Generator objects to be chained sequentially.
            constraints: List of Constraint objects to evaluate sequences.
            constraint_weights: List of weights for each constraint. If None, all weights are 1.0.
            num_steps: Number of optimization steps per sample() call.
            temperature: Maximum temperature for annealing.
            temperature_min: Minimum temperature for annealing.
            track_step_size: Progress tracking interval.
            custom_logging: Custom logging function that takes (step, sequences) arguments.
            verbose: Whether to print progress information.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
        )
        self.num_steps: int = num_steps
        self.temperature: float = temperature
        self.temperature_min: float = temperature_min
        self.track_step_size: int = track_step_size
        self.custom_logging: Optional[
            Callable[[int, Tuple[Segment, ...]], None]
        ] = custom_logging
        self.verbose: bool = verbose

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for SequentialGenerator.

        Raises:
            ValueError: If generators have different batch sizes or temperature parameters are invalid.
        """
        super()._validate_generator()

        # Check that all batch sizes are the same
        batch_sizes = [gen.batch_size for gen in self.generators]
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"All generators must have the same batch_size. Found: {batch_sizes}"
            )

        # Validate temperature parameters
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be positive, got {self.temperature_min}"
            )
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

    def sample(self) -> None:
        """
        Execute sequential sampling with chained autoregressive generators.

        Each step: (1) applies all generators sequentially with chaining,
        (2) evaluates energy change, (3) accepts/rejects based on Metropolis-Hastings
        with temperature annealing.

        Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        # Initialize sequential states
        self.score_energy()
        current_best_energy = np.min(self.energy_scores)
        self.append_snapshot_to_history(step=0)

        # Execute sequential optimization steps
        for step in range(1, self.num_steps + 1):
            # Calculate temperature with proper annealing (step 1 = T_max, final step = T_min)
            if self.num_steps == 1:
                cur_temp = self.temperature
            else:
                cur_temp = self.temperature * (self.temperature_min / self.temperature) ** (
                    (step - 1) / (self.num_steps - 1)
                )

            # Execute single sequential step
            current_best_energy = self._execute_sequential_step(
                step, cur_temp, current_best_energy
            )

            # Track progress periodically
            if step % self.track_step_size == 0:
                self.append_snapshot_to_history(step=step)
        
        # Always capture final state if it wasn't already captured
        if self.num_steps % self.track_step_size != 0:
            self.append_snapshot_to_history(step=self.num_steps)

    def _execute_sequential_step(
        self, step: int, cur_temp: float, current_best_energy: float
    ) -> float:
        """
        Execute a single sequential step including chaining, evaluation, and acceptance decision.

        Args:
            step: Current step number.
            cur_temp: Current temperature for this step.
            current_best_energy: Current best energy value.

        Returns:
            Updated best energy value.
        """
        # 1. Store old sequences for potential revert
        old_sequences_by_gen = self._backup_sequences()

        # 2. Apply all generators sequentially with chaining
        self._sample_sequential_generators()

        # 3. Evaluate new energy
        self.score_energy()
        new_best_energy = np.min(self.energy_scores)

        # 4. Accept or reject proposal according to Metropolis-Hastings algorithm
        original_best_energy = current_best_energy  # Save original for logging
        current_best_energy, accept, alpha = self._accept_or_reject_proposal(
            current_best_energy,
            new_best_energy,
            cur_temp,
            old_sequences_by_gen,
        )

        # 5. Log progress
        if self.verbose and step % self.track_step_size == 0:
            self._log_step(
                step, original_best_energy, new_best_energy, alpha, accept, cur_temp
            )

        return current_best_energy

    def _backup_sequences(self) -> List[List[Any]]:
        """
        Create backup copies of all sequences from all generators.

        Returns:
            List of backed up sequences organized by generator.
        """
        old_sequences_by_gen = []
        for generator in self.generators:
            gen_old_seqs = []
            for sequence_batch in generator.get_generator_outputs():
                for program_seq in sequence_batch:
                    gen_old_seqs.append(copy.deepcopy(program_seq))
            old_sequences_by_gen.append(gen_old_seqs)
        return old_sequences_by_gen

    def _sample_sequential_generators(self) -> None:
        """
        Apply all generators sequentially, chaining outputs between them.

        Each generator uses the accumulated output from previous generators
        as prompts for its own generation.
        """
        first_gen = self.generators[0]

        # Initialize running_prompts based on the first generator type
        if hasattr(first_gen, 'prompt_seqs'):
            # For generators that accept prompts
            running_prompts = first_gen.prompt_seqs.copy()
        else:
            # For generators that don't accept prompts
            outputs = first_gen.get_generator_outputs()
            if outputs:
                batch = outputs[0]
                running_prompts = [seq.sequence for seq in batch.batch_sequences]
            else:
                running_prompts = [""] * first_gen.batch_size

        # Sample from each generator in sequence, chaining outputs
        for i, generator in enumerate(self.generators):
            # For generators that accept prompts
            if self._is_extension_based_generator(generator):
                prompt_seqs = running_prompts if i > 0 else None
                generator.sample(prompt_seqs=prompt_seqs)
            else:
                # For generators that don't accept prompts
                generator.sample()

            # Accumulate this generator's output
            outputs = generator.get_generator_outputs()
            if len(outputs) != 1:
                raise ValueError(
                    f"Generator {i} must output exactly one Segment for chaining"
                )
            batch = outputs[0]

            # Update running_prompts with the generator's output
            if hasattr(generator, 'prompt_seqs') or i == 0:
                for batch_idx in range(len(batch)):
                    if i == 0 and getattr(generator, "prepend_prompt", False):
                        # First generator with prepend_prompt: output already includes prompt content,
                        # just add back the prefix tokens that were stripped
                        original_prompt = running_prompts[batch_idx] if hasattr(generator, 'prompt_seqs') else ""
                        generated = batch[batch_idx].sequence
                        valid_chars = batch._valid_chars or set()
                        prefix_tokens = "".join(
                            c for c in original_prompt if c not in valid_chars
                        )
                        running_prompts[batch_idx] = prefix_tokens + generated
                    else:
                        # Normal case: accumulate output to running prompts
                        running_prompts[batch_idx] += batch[batch_idx].sequence
            else:
                # For generators that don't accept prompts
                running_prompts = [seq.sequence for seq in batch.batch_sequences]

    def _accept_or_reject_proposal(
        self,
        current_best_energy: float,
        new_best_energy: float,
        cur_temp: float,
        old_sequences_by_gen: List[List[Any]],
    ) -> Tuple[float, bool, float]:
        """
        Compute Metropolis-Hastings acceptance probability and make decision.

        Args:
            current_best_energy: Energy of current best sequence.
            new_best_energy: Energy of proposed sequence.
            cur_temp: Current temperature for acceptance calculation.
            old_sequences_by_gen: Backup of sequences before proposal.
            new_energies: All energy values for the new sequences.

        Returns:
            Tuple of (updated_best_energy, accept, alpha).
        """
        # Compute acceptance probability
        energy_diff = -(new_best_energy - current_best_energy) / cur_temp
        energy_diff = min(energy_diff, MAX_EXP_ARG)  # Clamp to prevent overflow
        alpha = np.exp(energy_diff)
        alpha = min(1.0, alpha)
        accept = random.random() < alpha

        # Execute the decision
        if accept:
            # Accept: copy best sequences to all positions
            new_best_idx = np.argmin(self.energy_scores)
            self._replicate_best_sequence(new_best_idx)
            return new_best_energy, accept, alpha
        else:
            # Revert changes if rejected
            for i, generator in enumerate(self.generators):
                seq_idx = 0
                for sequence_batch in generator.get_generator_outputs():
                    for program_seq in sequence_batch:
                        program_seq.sequence = old_sequences_by_gen[i][seq_idx].sequence
                        program_seq._metadata = old_sequences_by_gen[i][
                            seq_idx
                        ]._metadata.copy()
                        seq_idx += 1
            return current_best_energy, accept, alpha

    def _log_step(
        self,
        step: int,
        old_energy: float,
        new_energy: float,
        alpha: float,
        accept: bool,
        cur_temp: float,
    ) -> None:
        """
        Log information about the current sequential generation step.

        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            cur_temp: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {cur_temp:.6f}, "
            f"accept: {accept}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()

    def _is_extension_based_generator(self, generator) -> bool:
        """
        Determine if a generator is extension-based or mutation-based.
        
        Args:
            generator: The generator to check
            
        Returns:
            True if the generator is extension-based, False if mutation-based
        """
        # Extension-based generators have prepend_prompt attribute
        # Mutation-based generators don't have this attribute
        return hasattr(generator, 'prepend_prompt') and generator.prepend_prompt

