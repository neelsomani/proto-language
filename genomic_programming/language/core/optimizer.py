"""
Optimizer base class for the proto-language.

Base class for iterative optimization algorithms that coordinate multiple
generators and constraints to search for optimal biological sequences.
"""

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .constraint import Constraint
from .construct import Construct
from .generator import Generator


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    Coordinates multiple generators and constraints to search for optimal
    biological sequences through iterative optimization. Unlike generators
    which modify sequences directly, optimizers orchestrate the search process
    by coordinating generators, evaluating constraints, and making decisions
    about which sequences to keep.
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the Optimizer.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            batch_size: Number of sequence variants to generate simultaneously.
        """
        self.batch_size = batch_size
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.history: List[Dict[str, Any]] = []  # Each entry: {"time_step": int, "energy_scores": List[float], "constructs": List[Construct]}
        self.energy_scores: List[float] = []  # Each index corresponds to a batch element, empty until first score_energy() call

        # Any batch_size specified during sub-generator construction is overwritten
        for i, gen in enumerate(self.generators):
            if gen.batch_size != self.batch_size:
                # Only warn if the generator was explicitly initialized with a non-default batch_size
                if gen.batch_size != 1:
                    warnings.warn(
                        f"Generator {i} ({gen.__class__.__name__}) was initialized with batch_size={gen.batch_size}, "
                        f"but Optimizer is overwriting it to batch_size={self.batch_size}. "
                        f"To avoid this warning, do not specify batch_size when creating sub-generators for Optimizer.",
                        UserWarning,
                        stacklevel=2
                    )
                # Expand the generator's assigned segment(s) to match the new batch_size
                for segment in gen.get_generator_outputs():
                    if segment.batch_size != self.batch_size:
                        segment.create_batch(self.batch_size)
            gen.batch_size = self.batch_size

        # Set self._generator_outputs to be a flat tuple of all Segment objects from all sub-generators
        self._generator_outputs = tuple(
            seq for gen in self.generators for seq in gen.get_generator_outputs()
        )  # Unused
        self._is_initialized = True

    def _validate_optimizer(self) -> None:
        """
        Validate that constructs, generators, constraints are properly configured.
        Must be called in final subclass __init__ to ensure all attributes are set.

        Raises:
            ValueError: If any validation checks fail.
        """

        # Ensure constructs, generators, and constraints are non-empty lists
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        if not self.generators:
            raise ValueError("Generators list cannot be empty")
        if not self.constraints:
            raise ValueError("Constraints list cannot be empty")

        # Ensure constraint_weights are positive and finite
        invalid_weights = [
            w for w in self.constraint_weights if w <= 0 or not np.isfinite(w)
        ]
        if invalid_weights:
            raise ValueError(
                f"Constraint weights must be positive and finite. Found invalid weights: {invalid_weights}"
            )

        # Ensure constraint count matches weight count
        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError(
                f"Constraint count ({len(self.constraints)}) must match weight count ({len(self.constraint_weights)})"
            )

        # Ensure types for all constructs, generators, and constraints are correct
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise ValueError(
                    f"Construct {i} has type {type(construct)}, expected Construct"
                )

        for i, generator in enumerate(self.generators):
            if not isinstance(generator, Generator):
                raise ValueError(
                    f"Generator {i} has type {type(generator)}, expected Generator"
                )

        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise ValueError(
                    f"Constraint {i} has type {type(constraint)}, expected Constraint"
                )

        # Ensure all generators are assigned construct segments
        if not all(generator._is_initialized for generator in self.generators):
            raise ValueError("Not all generators have been initialized.")

        # Ensure all construct segments are assigned to a generator
        unassigned_segments = [
            segment
            for construct in self.constructs
            for segment in construct.segments
            if not segment._is_assigned
        ]
        if unassigned_segments:
            raise ValueError(
                f"Found {len(unassigned_segments)} construct segments not assigned to any generator."
            )

        # Ensure all constraints have at least one generator-assigned input Segment
        generator_segment_ids = set(
            id(segment)
            for generator in self.generators
            for segment in generator.get_generator_outputs()
        )
        for i, constraint in enumerate(self.constraints):
            if not constraint.inputs:
                raise ValueError(f"Constraint {i} has no inputs assigned")
            if not any(id(inp) in generator_segment_ids for inp in constraint.inputs):
                raise ValueError(f"Constraint {i} has no generator-connected inputs")
# TODO: Figure out whether we need to remove this
    def get_generator_outputs(self):
        """
        Get all segments from all constructs as a flat tuple.

        Returns:
            Tuple of all Segment objects across all constructs.

        Note:
            This method flattens all segments from all constructs into a single tuple,
            which is useful for iterating over all segments being optimized.
        """
        return tuple(seg for construct in self.constructs for seg in construct.segments)

    def _replicate_best_sequence(self, best_idx: int) -> None:
        """
        Copy the best sequence to all positions within each Segment.

        This helper method ensures that when a proposal is accepted, the sequence
        with the best energy is propagated to all positions within each batch.
        This is essential for maintaining consistency in constructs access.

        Args:
            best_idx: Index of the best sequence to propagate across all batches.

        Raises:
            RuntimeError: If called before assign() has been called.
            ValueError: If any batch has fewer sequences than best_idx.

        Note:
            This method modifies sequences in-place for all generator outputs.
        """
        # Get generator outputs (works with both single and multiple outputs)
        generator_outputs = self.get_generator_outputs()

        for segment in generator_outputs:
            if len(segment.batch_sequences) <= best_idx:
                raise ValueError(
                    f"Segment has only {len(segment.batch_sequences)} sequences, "
                    f"cannot propagate best sequence at index {best_idx}"
                )

            best_sequence = segment.batch_sequences[best_idx]
            # Propagate the best sequence to all positions in this Segment
            # TODO: Check if this propagation makes sense for top_k implemenetation
            for sequence in segment.batch_sequences:
                sequence.sequence = best_sequence.sequence
                sequence._metadata = best_sequence._metadata.copy()

    @abstractmethod
    def sample(self, **kwargs: Any) -> None:
        """
        Run one or more steps of optimization.

        Subclasses should implement this method to run the optimization process.
        Implementations should modify generator outputs in-place and may store
        snapshots of constructs in `self.history`.

        Args:
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    def score_energy(self, operation: str = "add") -> None:
        """
        Compute energy scores by combining constraint evaluation scores
        Energy scores are stored in self.energy_scores.

        The energy function is computed as a weighted combination of all
        constraint scores. Lower energy values indicate better solutions.

        Args:
            operation: How to combine constraint scores across constraints:
                - 'add': Sum weighted constraint scores (default)
                - 'multiply': Multiply weighted constraint scores

        Raises:
            ValueError: If optimizer is not properly initialized or operation is not 'add' or 'multiply'.

        Note:
            Energy computation uses current sequence values, so it reflects
            the most recent state after any sampling operations. The computed
            energy scores are accessible via self.energy_scores.
        """
        # Ensure generator is properly initialized
        self._validate_optimizer()

        # Get weighted scores from all constraints: shape (n_constraints, n_samples)
        constraint_scores = np.array(
            [
                np.array(constraint.evaluate()) * weight
                for constraint, weight in zip(self.constraints, self.constraint_weights)
            ]
        )

        # Combine across constraints for each sample
        if operation == "multiply":
            energies = np.prod(constraint_scores, axis=0)
        elif operation == "add":
            energies = np.sum(constraint_scores, axis=0)
        else:
            raise ValueError(f"Operation must be 'multiply' or 'add', got {operation}")

        energies_list = energies.tolist()
        self.energy_scores = energies_list
    
    def append_snapshot_to_history(self, step: int = 0) -> None:
        """Save snapshot of current construct state and energy scores to history."""
        # Store as structured history entry with separate metadata
        history_entry = {
            "time_step": step,
            "energy_scores": self.energy_scores.copy(),
            "constructs": copy.deepcopy(self.constructs)
        }
        
        self.history.append(history_entry)

