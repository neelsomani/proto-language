"""
Mcmc Generator

Extracted from generator.py for better code organization.
"""

from typing import Any, Callable, List, Optional, Tuple, Dict, final
import copy
import random
import sys

import numpy as np

from ..base import IterativeGenerator, Construct, Generator, Constraint, Sequence, Segment

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0


@final
class MCMCGenerator(IterativeGenerator):
    """
    Metropolis-Hastings MCMC generator for constraint-driven sequence optimization.

    This generator implements a Metropolis-Hastings sampling algorithm that uses
    multiple sub-generators as proposal distributions and constraints to define
    the energy function. It's designed for iterative sequence refinement where
    proposals are accepted or rejected based on energy improvements.

    The generator supports simulated annealing, multiple constraints with weights,
    and flexible sequence concatenation for complex multi-part designs.

    Examples:
        Basic MCMC optimization:
        >>> constructs = [Construct([segment1, segment2])]
        >>> mcmc = MCMCGenerator(
        ...     constructs=constructs,
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     constraint_weights=[1.0, 2.0],  # Weight homopolymer constraint more
        ...     num_steps=100,
        ...     temperature=0.5,  # More greedy sampling
        ...     temperature_min=0.001
        ... )
        >>> mcmc.sample()
        >>> final_constructs = mcmc.constructs
        
        Top-k MCMC optimization:
        >>> mcmc_topk = MCMCGenerator(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],  # batch_size=10
        ...     constraints=[energy_constraint],
        ...     num_steps=100,
        ...     temperature=1.0,
        ...     top_k=5,  # Maintain 5 parent sequences
        ... )
        >>> # Each step generates 10 proposals per parent (50 total proposals)
        >>> # Applies MCMC acceptance (rejected proposals keep parent)
        >>> # Then selects top-5 by energy for next iteration
        >>> mcmc_topk.sample()
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
        top_k: int = 1,
        custom_logging: Optional[Callable[[int, Sequence], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MCMC generator with sub-generators and constraints.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects to generate sequences.
            constraints: List of Constraint objects to evaluate sequences.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            num_steps: Number of MCMC steps per sample() call.
            temperature: Maximum temperature for annealing.
            temperature_min: Minimum temperature for annealing.
            track_step_size: Interval for progress tracking.
            custom_logging: Custom logging function that takes (step, sequences) arguments.
            verbose: Whether to print progress information.
            top_k: Number of top sequences to maintain across iterations. When top_k=1 (default),
                   behaves like standard MCMC. When top_k>1, maintains k parent sequences and
                   generates batch_size proposals per parent each step (total: k x batch_size proposals).
                   Must be ≤ batch_size to ensure unique initial parent sequences.
                   Note: Generator batch sizes will be expanded to batch_size x k during sampling.

        Raises:
            ValueError: If any validation checks fail.
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
        self.top_k: int = top_k
        self.custom_logging: Optional[
            Callable[[int, Tuple[Segment, ...]], None]
        ] = custom_logging
        self.verbose: bool = verbose

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for MCMCGenerator.

        Raises:
            ValueError: If temperature parameters or top_k are invalid.
        """
        super()._validate_generator()

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
        
        # Validate top_k parameter
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        
        # Validate top_k <= batch_size
        initial_batch_size = self.generators[0].batch_size
        if self.top_k > initial_batch_size:
            raise ValueError(
                f"top_k ({self.top_k}) cannot be greater than batch_size ({initial_batch_size}). "
                f"top_k must be ≤ batch_size to ensure unique initial parent sequences."
            )

    def sample(self) -> None:
        """
        Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Maintains top_k parent sequences (k=1 for standard MCMC)
        2. Generates batch_size proposals per parent (total: k x batch_size proposals)
        3. Evaluates all proposal and applies MCMC acceptance: accepted proposals + rejected parents form candidate pool
        4. Selects top-k by energy from all candidates in candidate pool
        5. Optionally logs progress and tracks state

        Algorithm:
        - When batch_size=1 and top_k=1: behaves as standard single-chain MCMC
        - When top_k>1: maintains multiple parent sequences for diversity
        - For each proposal, applies Metropolis-Hastings acceptance criterion
        - Rejected proposals restore to parent state (deepcopied Sequence + energy)
        - After all proposals, selects top-k by energy for next iteration
        
        Note:
            - Simulated annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
            - batch_size means "proposals per parent" when top_k > 1
            - Total proposals per step: top_k x batch_size
            - Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        # Initialize: expand batches and select initial parents
        proposals_per_parent = self._initialize_topk()
        top_k_idx, parent_energies = self._get_initial_parents()
        self.append_snapshot_to_history(step=0)
        
        # Main MCMC loop
        for step in range(1, self.num_steps + 1):

            # Calculate temperature for simulated annealing
            cur_temp = self._calculate_temperature(step)
            
            # Save parent states before mutation (needed for rejection handling)
            parent_states = self._save_parent_states(top_k_idx)
            
            # Generate proposals from each parent
            self._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
            self._generate_proposals()
            
            # Apply MCMC acceptance and select top-k
            top_k_idx, parent_energies = self._select_topk_with_mcmc(
                top_k_idx, parent_energies, proposals_per_parent, cur_temp, parent_states
            )
            
            # Logging and history tracking
            if self._should_track_history(step):
                if self.verbose:
                    self._log_topk_progress(step, parent_energies)
                self.append_snapshot_to_history(step=step)
        
        self._track_final_state_if_needed()
    
    def _initialize_topk(self) -> int:
        """Initialize batch sizes and score initial energies.
        
        Expands generator and constraint batch sizes to (proposals_per_parent x top_k).
        When top_k=1, this is a no-op (1 x batch_size = batch_size).
        
        Returns:
            proposals_per_parent: Number of proposals to generate per parent sequence.
        """
        proposals_per_parent = self.generators[0].batch_size
        expanded_batch_size = proposals_per_parent * self.top_k
        
        if self.verbose:
            print(f"Top-k MCMC initialization:")
            print(f"  top_k={self.top_k}, proposals_per_parent={proposals_per_parent}")
            print(f"  expanded_batch_size={expanded_batch_size}")
        
        for gen in self.generators:
            gen.batch_size = expanded_batch_size
        
        for segment in self.get_generator_outputs():
            segment.create_batch(expanded_batch_size)
        
        for constraint in self.constraints:
            constraint.batch_size = expanded_batch_size
        
        self.score_energy()
        return proposals_per_parent
    
    def _get_initial_parents(self) -> Tuple[np.ndarray, List[float]]:
        """Select initial top-k parent sequences by energy.
        
        Returns:
            top_k_idx: Indices of the k best sequences (k=1 for standard MCMC).
            parent_energies: Energy values of the selected parent sequences.
        """
        top_k_idx = np.argsort(self.energy_scores)[:self.top_k]
        parent_energies = [self.energy_scores[i] for i in top_k_idx]
        
        if self.verbose:
            print(f"  Initial parent energies: {[f'{e:.4f}' for e in parent_energies]}")
            print()
        
        return top_k_idx, parent_energies
    
    def _replicate_parents_to_batch(self, top_k_idx: np.ndarray, proposals_per_parent: int) -> None:
        """Replicate each parent sequence to its designated batch positions.
        
        Each parent is copied to a contiguous block of batch positions:
        - Parent 0: positions [0, proposals_per_parent)
        - Parent 1: positions [proposals_per_parent, 2*proposals_per_parent)
        - etc.
        
        When top_k=1, this copies sequence[0] to itself (no-op effect).
        
        Args:
            top_k_idx: Indices of parent sequences to replicate.
            proposals_per_parent: Number of copies per parent.
        """
        for parent_pos, parent_idx in enumerate(top_k_idx):
            start_idx = parent_pos * proposals_per_parent
            end_idx = (parent_pos + 1) * proposals_per_parent
            # Deepcopy parent to all positions in this parent's range for independence
            for segment in self.get_generator_outputs():
                source_seq = segment.batch_sequences[parent_idx]
                for idx in range(start_idx, end_idx):
                    segment.batch_sequences[idx] = copy.deepcopy(source_seq)
    
    def _save_parent_states(self, top_k_idx: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Save complete parent states before mutation using deepcopy.
        
        This is critical for the MCMC rejection mechanism: if a proposal is rejected,
        we restore the parent's complete state (Sequence object + energy).
        
        Args:
            top_k_idx: Indices of parent sequences to save.
            
        Returns:
            Dict mapping parent_idx -> {
                'segments': {segment_id -> deepcopied Sequence object},
                'energy': float
            }
        """
        parent_states = {}
        for parent_idx in top_k_idx:
            parent_states[parent_idx] = {
                'segments': {},
                'energy': self.energy_scores[parent_idx]
            }
            for segment in self.get_generator_outputs():
                seg_id = id(segment)
                # Deepcopy captures complete Sequence state (sequence, metadata, all attributes)
                parent_states[parent_idx]['segments'][seg_id] = copy.deepcopy(
                    segment.batch_sequences[parent_idx]
                )
        return parent_states
    
    def _restore_parent_state(
        self,
        target_idx: int,
        parent_idx: int,
        parent_states: Dict[int, Dict[str, Any]]
    ) -> None:
        """Restore complete parent state after MCMC rejection.
        
        Replaces the rejected proposal at target_idx with the deepcopied parent
        Sequence object and restores its energy score.
        
        Args:
            target_idx: Batch index to restore to (the rejected proposal's position).
            parent_idx: Index of the parent in the saved states dict.
            parent_states: Dict from _save_parent_states() containing deepcopied
                Sequence objects and energy scores.
        """
        # Deepcopy again to ensure independent Sequence objects at each batch position
        for segment in self.get_generator_outputs():
            seg_id = id(segment)
            segment.batch_sequences[target_idx] = copy.deepcopy(
                parent_states[parent_idx]['segments'][seg_id]
            )
        
        # Restore energy score to match the restored sequence
        self.energy_scores[target_idx] = parent_states[parent_idx]['energy']
    
    def _generate_proposals(self) -> None:
        """Generate proposals by sampling from a random generator and scoring energies.
        
        Picks one generator randomly, calls its sample() method to mutate sequences,
        then evaluates all sequences using constraints to compute energy scores.
        """
        generator = random.choice(self.generators)
        generator.sample()
        self.score_energy()
    
    def _select_topk_with_mcmc(
        self,
        top_k_idx: np.ndarray,
        parent_energies: List[float],
        proposals_per_parent: int,
        temperature: float,
        parent_states: Dict[int, Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[float]]:
        """Apply Metropolis-Hastings acceptance criterion and select top-k sequences.
        
        For each proposal:
        1. Compute acceptance probability: alpha = min(1, exp(-(E_proposal - E_parent) / T))
        2. Accept with probability alpha: keep proposal
        3. Reject with probability (1-alpha): restore parent state (sequence, metadata, energy)
        4. Add either accepted proposal or restored parent to candidates
        
        After all proposals are evaluated, select top-k candidates by energy.
        When top_k=1, this behaves as standard single-chain MCMC.
        
        Args:
            top_k_idx: Indices of current parent sequences.
            parent_energies: Energy values of current parents.
            proposals_per_parent: Number of proposals generated per parent.
            temperature: Current temperature for acceptance calculation.
            parent_states: Saved parent states (sequences, metadata, energies).
            
        Returns:
            new_top_k_idx: Indices of selected top-k sequences for next iteration.
            new_parent_energies: Energy values of selected sequences.
        """
        candidates = []
        
        for parent_pos, (parent_idx, parent_energy) in enumerate(zip(top_k_idx, parent_energies)):
            start_idx = parent_pos * proposals_per_parent
            end_idx = (parent_pos + 1) * proposals_per_parent
            
            for proposal_idx in range(start_idx, end_idx):
                proposal_energy = self.energy_scores[proposal_idx]
                alpha = self._compute_acceptance_prob(parent_energy, proposal_energy, temperature)
                
                if random.random() < alpha:
                    # Accept proposal
                    candidates.append((proposal_energy, proposal_idx))
                else:
                    # Reject - restore complete parent state to this position
                    self._restore_parent_state(proposal_idx, parent_idx, parent_states)
                    # Use the saved parent energy for candidate list
                    restored_energy = parent_states[parent_idx]['energy']
                    candidates.append((restored_energy, proposal_idx))
        
        candidates.sort(key=lambda x: x[0])
        top_k_candidates = candidates[:self.top_k]
        
        new_top_k_idx = np.array([idx for _, idx in top_k_candidates])
        new_parent_energies = [energy for energy, _ in top_k_candidates]
        
        return new_top_k_idx, new_parent_energies
    
    def _log_topk_progress(self, step: int, parent_energies: List[float]) -> None:
        """Log optimization progress.
        
        Prints current step, energy statistics, and temperature for monitoring.
        When top_k=1, only best energy is meaningful (mean=best, std=0).
        
        Args:
            step: Current MCMC iteration number.
            parent_energies: Energy values of current top-k sequences.
        """
        best_energy = min(parent_energies)
        mean_energy = np.mean(parent_energies)
        worst_energy = max(parent_energies)
        std_energy = np.std(parent_energies) if len(parent_energies) > 1 else 0.0
        current_temp = self._calculate_temperature(step)
        
        # Format output based on top_k
        if self.top_k == 1:
            print(
                f"Iteration {step:4d} | "
                f"energy: {best_energy:.6f}, "
                f"T: {current_temp:.4f}"
            )
        else:
            print(
                f"Iteration {step:4d} | "
                f"best: {best_energy:.6f}, "
                f"mean: {mean_energy:.6f}, "
                f"worst: {worst_energy:.6f}, "
                f"std: {std_energy:.6f}, "
                f"T: {current_temp:.4f}"
            )
        
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()
    
    # ==================== Utility Helper Methods ====================
    
    def _calculate_temperature(self, step: int) -> float:
        """Calculate annealed temperature for given step.
        
        Uses exponential cooling schedule: T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))

        For num_steps=1, T(step) = T_max
        For num_steps>1, T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))
        For step=num_steps, T(step) = T_min
        
        Note: -1 is required in denominator because we loop from 1 to num_steps in the MCMC sampling loop
        
        Args:
            step: Current optimization step (1 to num_steps).
            
        Returns:
            Current temperature for this step.
        """

        # Handle division by 0 for num_steps=1
        if self.num_steps == 1:
            return self.temperature
        return self.temperature * (self.temperature_min / self.temperature) ** ((step - 1) / (self.num_steps - 1))
    
    def _compute_acceptance_prob(self, current_energy: float, proposed_energy: float, temperature: float) -> float:
        """Compute Metropolis-Hastings acceptance probability."""
        energy_diff = -(proposed_energy - current_energy) / temperature
        energy_diff = min(energy_diff, MAX_EXP_ARG)
        return min(1.0, np.exp(energy_diff))
    
    def _should_track_history(self, step: int) -> bool:
        """Check if current step should be tracked in history."""
        return step % self.track_step_size == 0
    
    def _track_final_state_if_needed(self) -> None:
        """Save final state to history if it wasn't already tracked."""
        if self.num_steps % self.track_step_size != 0:
            self.append_snapshot_to_history(step=self.num_steps)

