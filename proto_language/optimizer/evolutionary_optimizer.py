"""Evolutionary algorithm optimizer with population-based search, crossover, and selection.

Provides the ``evolutionary`` optimization strategy: a population-based algorithm that
maintains N candidate sequences across generations. Each generation produces offspring through
crossover (recombining parent sequences) and mutation (using the framework's existing mutation
generators), evaluates all offspring against the constraints, and selects survivors via
tournament selection with optional elitism. Unlike single-chain MCMC which explores one basin
at a time, the population maintains diversity across multiple promising regions simultaneously.

Examples:
    >>> from proto_language.constraint import gc_content_constraint
    >>> from proto_language.core import Constraint, Construct, Program, Segment
    >>> from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
    >>> from proto_language.optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerConfig
    >>> seg = Segment(length=20, sequence_type="dna")
    >>> gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    >>> gen.assign(seg)
    >>> gc = Constraint(inputs=[seg], function=gc_content_constraint, function_config={"min_gc": 80, "max_gc": 90})
    >>> optimizer = EvolutionaryOptimizer(
    ...     constructs=[Construct([seg])],
    ...     generators=[gen],
    ...     constraints=[gc],
    ...     config=EvolutionaryOptimizerConfig(population_size=10, num_generations=20),
    ... )
    >>> Program(optimizers=[optimizer], num_results=10).run()
"""

import copy
import logging
import math
from collections.abc import Callable
from typing import Any, Literal, final

from pydantic import model_validator

from proto_language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Sequence,
)
from proto_language.optimizer.optimizer_registry import optimizer
from proto_language.utils.base import BaseOptimizerConfig, ConfigField

logger = logging.getLogger(__name__)


class EvolutionaryOptimizerConfig(BaseOptimizerConfig):
    """Configuration object for EvolutionaryOptimizer.

    This class defines configuration parameters for the evolutionary algorithm optimizer,
    which maintains a population of candidate sequences and evolves them through
    crossover, mutation, and selection over multiple generations.

    Attributes:
        population_size (int): Number of individuals in the population. Each individual
            is a candidate sequence that evolves across generations. Must be at least 2
            (minimum for crossover) and divisible by tournament_size for balanced
            tournaments. Overrides program-level ``num_results`` if set.

        num_generations (int): Number of evolutionary generations to run. Each generation
            produces offspring through crossover and mutation, evaluates them, and selects
            survivors. More generations allow better convergence but increase runtime.
            Must be at least 1.

        elitism_count (int): Number of best individuals (lowest energy) to carry unchanged
            into the next generation. Elitism ensures the best solution never regresses
            across generations. Must be non-negative and less than ``population_size``.
            Default: 1.

        tournament_size (int): Number of individuals competing in each tournament selection.
            Larger tournaments create stronger selection pressure (favor better individuals
            more heavily). Must be at least 2 and at most ``population_size``. Default: 3.

        crossover_rate (float): Probability of applying crossover to create an offspring.
            When crossover occurs, two parents are selected and their genetic material is
            combined. When crossover does not occur (1 - crossover_rate), a single parent
            is cloned. Must be in [0.0, 1.0]. Default: 0.8.

        mutation_rate (float): Probability of applying mutation to each offspring after
            crossover. Mutation uses the assigned generators to modify sequences. Must be
            in [0.0, 1.0]. Default: 0.2.

        crossover_strategy (Literal["single-point", "uniform"]): Strategy for combining
            parent sequences:
            - ``"single-point"``: Select one random position and swap segments before/after
            - ``"uniform"``: For each position, randomly choose which parent contributes
            Default: ``"single-point"``.

        verbose (bool): Whether to print detailed progress information including
            population statistics and diversity metrics at each generation. Default: ``False``.

        tracking_interval (int): Number of generations between progress snapshots.
        track_proposals (bool): Whether to record offspring sequences alongside population.

    Note:
        - ``population_size`` determines ``num_results`` (one result per population member)
        - Total evaluations per generation: ``population_size`` offspring created and scored
        - Best practice: ``elitism_count`` >= 1 to preserve best solutions
    """

    # Required parameters
    population_size: int = ConfigField(
        ge=2,
        title="Population Size",
        description="Number of individuals in the population; each is a candidate sequence. Overrides program count.",
    )
    num_generations: int = ConfigField(
        ge=1,
        title="Number of Generations",
        description="Number of evolutionary generations. Each generates offspring via crossover/mutation and selects survivors.",
    )

    # num_results is managed internally (equals population_size)
    # Base optimizer uses this field; it's set in the validator
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Number of Results",
        description="Internal field automatically set to population_size. Do not set directly; use population_size.",
    )

    # Selection and elitism
    elitism_count: int = ConfigField(
        default=1,
        ge=0,
        title="Elitism Count",
        description="Top individuals (lowest energy) carried unchanged to next generation; ensures best never regresses.",
    )
    tournament_size: int = ConfigField(
        default=3,
        ge=2,
        title="Tournament Size",
        description="Individuals per tournament; larger values increase selection pressure favoring better individuals.",
    )

    # Genetic operators
    crossover_rate: float = ConfigField(
        default=0.8,
        ge=0.0,
        le=1.0,
        title="Crossover Rate",
        description="Probability of crossover (combining two parents); otherwise clone single parent.",
    )
    mutation_rate: float = ConfigField(
        default=0.2,
        ge=0.0,
        le=1.0,
        title="Mutation Rate",
        description="Probability of mutating each offspring using the assigned generators.",
    )
    crossover_strategy: Literal["single-point", "uniform"] = ConfigField(
        default="single-point",
        title="Crossover Strategy",
        description="Recombination method: 'single-point' swaps at one position, 'uniform' per-position random choice.",
    )

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "EvolutionaryOptimizerConfig":
        """Validate cross-field constraints and sync num_results with population_size."""
        if self.elitism_count >= self.population_size:
            raise ValueError(
                f"elitism_count ({self.elitism_count}) must be < population_size ({self.population_size})"
            )
        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) cannot exceed population_size ({self.population_size})"
            )
        # Sync num_results to population_size (base optimizer will use num_results)
        object.__setattr__(self, "num_results", self.population_size)
        return self


@optimizer(
    key="evolutionary",
    label="Evolutionary Optimizer",
    config=EvolutionaryOptimizerConfig,
    description="Population-based evolutionary algorithm: maintains diverse candidate sequences, recombines them through crossover, mutates offspring using the generators, and selects survivors via tournament selection with elitism to preserve the best solutions across generations.",
)
@final
class EvolutionaryOptimizer(Optimizer):
    """Evolutionary algorithm optimizer for population-based sequence optimization.

    This optimizer implements a genetic algorithm that maintains a population of candidate
    sequences and evolves them over generations through crossover (recombination), mutation
    (using framework generators), and tournament selection. Unlike MCMC which follows a single
    trajectory, the EA maintains population diversity and explores multiple promising regions
    simultaneously.

    Each generation:
    1. Selects elite individuals (best ``elitism_count`` by energy) to survive unchanged
    2. Fills remaining population through tournament selection and genetic operators:
       - Select parents via tournament (``tournament_size`` individuals compete)
       - Apply crossover with probability ``crossover_rate`` to combine two parents
       - Apply mutation with probability ``mutation_rate`` using assigned generators
    3. Evaluates all offspring against constraints
    4. Forms next generation from elites + selected offspring

    Attributes:
        population_size: Number of individuals in the population.
        num_generations: Total generations to evolve.
        elitism_count: Best individuals preserved unchanged per generation.
        tournament_size: Individuals competing in each tournament.
        crossover_rate: Probability of crossover operation.
        mutation_rate: Probability of mutation operation.
        crossover_strategy: Method for recombining parents ("single-point" or "uniform").

    Example:
        >>> config = EvolutionaryOptimizerConfig(population_size=20, num_generations=50, elitism_count=2)
        >>> optimizer = EvolutionaryOptimizer(
        ...     constructs=constructs, generators=[mutation_gen], constraints=[gc_constraint], config=config
        ... )
        >>> optimizer.run()
        >>> best_sequence = optimizer.constructs[0].segments[0].result_sequences[0]

    Note:
        - Crossover requires sequences of the same length across all segments
        - Mutation reuses framework generators (no new mutation primitive needed)
        - Tournament selection: randomly sample ``tournament_size`` individuals, pick best
        - Elitism guarantees monotonic improvement of the best individual's energy
    """

    # Class attribute required by OptimizerRegistry
    config_class = EvolutionaryOptimizerConfig
    config: EvolutionaryOptimizerConfig

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: EvolutionaryOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Evolutionary Optimizer.

        Args:
            constructs (list[Construct]): List of Construct objects to optimize.
            generators (list[Generator]): List of Generator objects for mutation operations.
            constraints (list[Constraint]): List of Constraint objects for fitness evaluation.
            config (EvolutionaryOptimizerConfig): Configuration object containing algorithm parameters.
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked generations (governed by ``tracking_interval``).
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If any validation checks fail.
        """
        self.config = config

        # Population size determines num_results (one result per population member)
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_results=config.population_size,
            proposals_per_result=1,  # Each individual produces offspring 1:1
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )

        self.population_size: int = config.population_size
        self.num_generations: int = config.num_generations
        self.elitism_count: int = config.elitism_count
        self.tournament_size: int = config.tournament_size
        self.crossover_rate: float = config.crossover_rate
        self.mutation_rate: float = config.mutation_rate
        self.crossover_strategy: str = config.crossover_strategy

        # Override base class num_steps for progress tracking
        self.num_steps = self.num_generations

    def run(self) -> None:
        """Execute evolutionary algorithm for sequence optimization.

        Runs the specified number of generations, where each generation:
        1. Evaluates current population's fitness (constraint-based energy scores)
        2. Selects elite individuals (best ``elitism_count``) to preserve
        3. Creates offspring via tournament selection, crossover, and mutation
        4. Forms next generation from elites + offspring

        The population is maintained in ``result_sequences`` across generations.
        Energy scores track fitness (lower is better).

        Note:
            - Initial population comes from upstream optimizer or original sequences
            - Elitism ensures best energy score never increases across generations
            - Diversity is maintained through population-based search
            - Snapshots of population at tracked generations are stored in self.history
        """
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing

        n_filter = sum(1 for c in self.constraints if c.threshold is not None)
        n_score = len(self.constraints) - n_filter
        logger.info(
            f"EvolutionaryOptimizer: {self.num_generations} generations, "
            f"population_size={self.population_size}, elitism={self.elitism_count}, "
            f"tournament_size={self.tournament_size}, crossover_rate={self.crossover_rate:.2f}, "
            f"mutation_rate={self.mutation_rate:.2f}, {len(self.constraints)} constraints "
            f"({n_filter} filter, {n_score} scoring)"
        )

        # Evaluate initial population
        self._sync_population_to_proposals()
        self.score_energy()
        self._sync_proposals_to_population()

        logger.debug(f"EvolutionaryOptimizer initial best energy: {min(self.energy_scores):.4f}")

        # Track initial state
        self._save_progress_snapshot(
            time_step=0,
            optimizer_metadata={
                "type": "evolutionary",
                "num_generations": self.num_generations,
                "population_size": self.population_size,
                "elitism_count": self.elitism_count,
                "tournament_size": self.tournament_size,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "crossover_strategy": self.crossover_strategy,
                "proposal_count": len(self._proposal_outcomes),
                "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
            },
        )

        # Evolutionary loop
        for generation in range(1, self.num_generations + 1):
            # 1. Select elites (best individuals by energy)
            elite_indices = self._select_elites()

            # Save elite energies before scoring offspring
            elite_energies = [self.energy_scores[idx] for idx in elite_indices]

            # 2. Create offspring to fill remaining population slots
            num_offspring = self.population_size - self.elitism_count
            self._create_offspring(num_offspring)

            # 3. Evaluate offspring
            # offspring are already in proposal_sequences from _create_offspring
            self.score_energy()
            offspring_energies = list(self.energy_scores)  # Save offspring energies

            # 4. Form next generation: elites + offspring
            self._update_population(elite_indices, elite_energies, offspring_energies)

            # Save snapshot and log at tracking interval or final generation
            if generation % self.tracking_interval == 0 or generation == self.num_generations:
                self._save_progress_snapshot(
                    time_step=generation,
                    optimizer_metadata={
                        "type": "evolutionary",
                        "num_generations": self.num_generations,
                        "population_size": self.population_size,
                        "elitism_count": self.elitism_count,
                        "tournament_size": self.tournament_size,
                        "crossover_rate": self.crossover_rate,
                        "mutation_rate": self.mutation_rate,
                        "crossover_strategy": self.crossover_strategy,
                        "proposal_count": len(self._proposal_outcomes),
                        "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
                    },
                )
                self._log_evolution_progress(generation)

    def _sync_population_to_proposals(self) -> None:
        """Copy current population (result_sequences) to proposal pool for evaluation."""
        for segment in self.segments:
            segment.proposal_sequences = [copy.deepcopy(seq) for seq in segment.result_sequences]

    def _sync_proposals_to_population(self) -> None:
        """Copy evaluated proposals back to population (result_sequences).

        Also syncs energy_scores to match the population.
        """
        for segment in self.segments:
            segment.result_sequences = [copy.deepcopy(seq) for seq in segment.proposal_sequences]
        # energy_scores is already correct from score_energy()

    def _select_elites(self) -> list[int]:
        """Select best individuals (lowest energy) to preserve unchanged.

        Returns:
            list[int]: Indices of elite individuals sorted by energy (best first).
        """
        # Sort population indices by energy score (ascending = best first)
        sorted_indices = sorted(range(self.population_size), key=lambda i: self.energy_scores[i])
        return sorted_indices[: self.elitism_count]

    def _tournament_selection(self) -> int:
        """Select one individual via tournament selection.

        Randomly samples ``tournament_size`` individuals and returns the index of the
        one with the lowest energy score.

        Returns:
            int: Index of the selected individual.
        """
        # Randomly sample tournament_size individuals without replacement
        tournament_indices = self._rng.sample(range(self.population_size), self.tournament_size)
        # Return index of best individual in tournament (lowest energy)
        return min(tournament_indices, key=lambda i: self.energy_scores[i])

    def _crossover(self, parent1_idx: int, parent2_idx: int, target_idx: int) -> None:
        """Apply crossover operator to create offspring from two parents.

        Modifies proposal_sequences[target_idx] in place by recombining parent1 and parent2 sequences.

        Args:
            parent1_idx (int): Index of first parent in result_sequences.
            parent2_idx (int): Index of second parent in result_sequences.
            target_idx (int): Index in proposal_sequences where offspring will be placed.
        """
        for segment in self.segments:
            parent1_seq = segment.result_sequences[parent1_idx].sequence
            parent2_seq = segment.result_sequences[parent2_idx].sequence

            if len(parent1_seq) != len(parent2_seq):
                raise RuntimeError(
                    f"Crossover requires equal-length sequences, got {len(parent1_seq)} and {len(parent2_seq)}"
                )

            if self.crossover_strategy == "single-point":
                # Single-point crossover: pick random position, swap before/after
                if len(parent1_seq) > 1:
                    point = self._rng.randint(1, len(parent1_seq) - 1)
                    child_seq = parent1_seq[:point] + parent2_seq[point:]
                else:
                    # Sequence too short for single-point crossover, randomly pick one parent
                    child_seq = parent1_seq if self._rng.random() < 0.5 else parent2_seq
            elif self.crossover_strategy == "uniform":
                # Uniform crossover: for each position, randomly pick parent
                child_seq = "".join(
                    parent1_seq[i] if self._rng.random() < 0.5 else parent2_seq[i] for i in range(len(parent1_seq))
                )
            else:
                raise ValueError(f"Unknown crossover strategy: {self.crossover_strategy}")

            # Create child sequence and place in proposal pool at target index
            segment.proposal_sequences[target_idx] = Sequence(
                sequence=child_seq,
                sequence_type=segment.sequence_type,
                valid_chars=segment.valid_chars,
            )

    def _create_offspring(self, num_offspring: int) -> list[int]:
        """Create offspring through selection, crossover, and mutation.

        Returns:
            list[int]: Indices in proposal_sequences where offspring were placed (after scoring).
        """
        # Pre-allocate proposal pool with correct size
        for segment in self.segments:
            segment.proposal_sequences = [
                Sequence(sequence="", sequence_type=segment.sequence_type, valid_chars=segment.valid_chars)
                for _ in range(num_offspring)
            ]

        # Track which offspring should be mutated
        offspring_to_mutate: list[int] = []

        # Phase 1: Create offspring through selection and crossover
        for i in range(num_offspring):
            # Selection: tournament selection for parent(s)
            parent1_idx = self._tournament_selection()

            # Crossover or cloning
            if self._rng.random() < self.crossover_rate:
                # Crossover: select second parent and recombine
                parent2_idx = self._tournament_selection()
                self._crossover(parent1_idx, parent2_idx, target_idx=i)
            else:
                # No crossover: clone parent1
                for segment in self.segments:
                    segment.proposal_sequences[i] = copy.deepcopy(segment.result_sequences[parent1_idx])

            # Decide whether this offspring will be mutated
            if self._rng.random() < self.mutation_rate:
                offspring_to_mutate.append(i)

        # Phase 2: Apply mutation to selected offspring
        # Process each offspring individually by temporarily resizing proposal pool
        if self.generators and offspring_to_mutate:
            # Save all offspring before mutations
            saved_offspring: dict[int, list[Sequence]] = {
                id(seg): [copy.deepcopy(seq) for seq in seg.proposal_sequences] for seg in self.segments
            }

            for idx in offspring_to_mutate:
                # Set proposal pool to just this one offspring
                for segment in self.segments:
                    segment.proposal_sequences = [saved_offspring[id(segment)][idx]]

                # Apply one random generator to mutate this offspring
                generator = self._rng.choice(self.generators)
                generator.sample()

                # Save the mutated result
                for segment in self.segments:
                    saved_offspring[id(segment)][idx] = segment.proposal_sequences[0]

            # Restore full proposal pool with all offspring (some mutated)
            for segment in self.segments:
                segment.proposal_sequences = saved_offspring[id(segment)]

        return list(range(num_offspring))

    def _update_population(
        self, elite_indices: list[int], elite_energies: list[float], offspring_energies: list[float]
    ) -> None:
        """Form next generation from elites and evaluated offspring.

        Args:
            elite_indices (list[int]): Indices of elites in current result_sequences.
            elite_energies (list[float]): Energy scores of elites (saved before offspring evaluation).
            offspring_energies (list[float]): Energy scores of offspring (from score_energy()).
        """
        # Build next generation per segment
        new_population_per_segment: dict[int, list[Sequence]] = {id(seg): [] for seg in self.segments}
        new_energies: list[float] = []

        # Add elites (from current population)
        for idx, energy in zip(elite_indices, elite_energies, strict=True):
            new_energies.append(energy)
            for segment in self.segments:
                new_population_per_segment[id(segment)].append(copy.deepcopy(segment.result_sequences[idx]))

        # Add offspring (from evaluated proposals)
        for idx, energy in enumerate(offspring_energies):
            new_energies.append(energy)
            for segment in self.segments:
                new_population_per_segment[id(segment)].append(copy.deepcopy(segment.proposal_sequences[idx]))

        # Update population
        for segment in self.segments:
            segment.result_sequences = new_population_per_segment[id(segment)]

        self.energy_scores = new_energies

    def _log_evolution_progress(self, generation: int) -> None:
        """Log optimization progress as a multi-line INFO block."""
        logger.info(f"Generation {generation}/{self.num_generations}")
        filter_summary = self._format_filter_summary()
        if filter_summary is not None:
            logger.info(f"  filters: {filter_summary}")
        for line in self._format_scoring_lines():
            logger.info(f"  {line}")
        logger.info(f"  energy:  {self._format_energy_summary()}")

        # Population diversity metrics
        finite_energies = [e for e in self.energy_scores if math.isfinite(e)]
        if finite_energies:
            unique_energies = len(set(finite_energies))
            logger.info(f"  diversity: {unique_energies}/{len(finite_energies)} unique fitness values")

        if self.custom_logging:
            self.custom_logging(generation, self.segments)
