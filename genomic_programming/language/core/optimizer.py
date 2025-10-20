"""
Optimizer base class for the proto-language.

Base class for iterative optimization algorithms that coordinate multiple
generators and constraints to search for optimal biological sequences.
"""

import copy
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
        num_candidates: int,
        num_selected: int,
        constraint_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize the Optimizer with dual-pool semantics.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            num_candidates: Number of candidate proposals to generate per iteration.
            num_selected: Number of sequences to select and maintain as results.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
        """
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.num_candidates = num_candidates
        self.num_selected = num_selected
        self.history: List[Dict[str, Any]] = []  # Each entry: {"time_step": int, "energy_scores": List[float], "constructs": List[Construct]}
        self.energy_scores: List[float] = []  # Each index corresponds to a candidate, empty until first score_energy() call

        # Set generator batch_size to num_candidates (generators write to candidate pool)
        for gen in self.generators:
            gen.batch_size = self.num_candidates

        self._is_initialized = True

    @property
    def segments(self):
        """All segments from all constructs being optimized."""
        return tuple(seg for construct in self.constructs for seg in construct.segments)

    @abstractmethod
    def sample(self) -> None:
        """
        Subclasses should implement this method to run the optimization process.
        Implementations should modify generator outputs in-place.
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

        # Call each generator's validation
        for generator in self.generators:
            generator._validate_generator()

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
