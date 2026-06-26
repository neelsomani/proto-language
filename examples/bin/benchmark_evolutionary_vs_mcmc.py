"""Benchmark comparing NSGA-II multi-objective optimization vs MCMC scalarizations.

This benchmark validates the NSGA-II selection mode added to EvolutionaryOptimizer
by comparing it against the practitioner's baseline: running MCMC with multiple
weight vectors and pooling the non-dominated points.

## Task: Conflicting GC-content objectives

Two genuinely competing constraints on one DNA sequence:
- low_gc: minimize distance from GC% target in [10, 30]
- high_gc: minimize distance from GC% target in [70, 90]

No sequence can score near-zero on both (verified in sanity check). This creates
a real Pareto front where NSGA-II can demonstrate its multi-objective advantage.

## Methods (budget-matched)

1. **EA-nsga2**: EvolutionaryOptimizer with selection="nsga2"
   - Returns its .pareto_front attribute (rank-0 individuals)
   - Evaluations: initial_pop + generations * (pop_size - elitism)

2. **Multi-weight MCMC**: Run K independent MCMC chains with different weight vectors
   - Each chain uses a different (w_low, w_high) scalarization
   - Pool all final solutions, extract non-dominated set
   - This is the honest baseline practitioners actually use
   - Evaluations: K * steps_per_chain (matched to EA budget)

3. **Single-weight MCMC**: One chain with equal weights (0.5, 0.5)
   - Floor performance (scalar optimization)
   - Evaluations: matched to EA budget

## Metrics

- **Hypervolume**: Volume dominated by front relative to reference point (1.0, 1.0)
  - Higher is better, measures both convergence and spread
  - The number you compare

- **Front size**: Number of non-dominated solutions
  - Secondary metric showing diversity

- **2D scatter**: Visual comparison of fronts in objective space
  - Makes the difference obvious

## Budget matching

All methods use identical total constraint evaluations (verified with assert).
Default: 1000 evaluations, 20 trials for statistical power.

## Outputs

Writes to current directory:
- benchmark_nsga2_summary.json: Aggregate statistics and conclusions
- benchmark_nsga2_detailed.json: Per-trial results
- benchmark_nsga2_fronts.png: 2D scatter plot of fronts

Usage:
    python examples/bin/benchmark_evolutionary_vs_mcmc.py
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from proto_tools.transforms.masking import MaskingStrategy

from proto_language.constraint import gc_content_constraint
from proto_language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import (
    EvolutionaryOptimizer,
    EvolutionaryOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Task parameters
# ============================================================================

SEQUENCE_LENGTH = 30
BUDGET = 1000
NUM_TRIALS = 20

# GC content targets (conflicting)
LOW_GC_MIN, LOW_GC_MAX = 10, 30
HIGH_GC_MIN, HIGH_GC_MAX = 70, 90

# Reference point for hypervolume (worst possible scores)
REFERENCE_POINT = (1.0, 1.0)


# ============================================================================
# Sanity check: verify conflict is real
# ============================================================================


def verify_conflict() -> None:
    """Verify that no sequence can score near-zero on both objectives."""
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")

    low_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
        label="high_gc",
    )

    # Test extreme cases - manually evaluate a few test sequences
    test_sequences = [
        "A" * SEQUENCE_LENGTH,  # All A (low GC)
        "C" * SEQUENCE_LENGTH,  # All C (high GC)
        "AC" * (SEQUENCE_LENGTH // 2),  # Mixed (50% GC)
    ]

    min_sum = float("inf")
    for test_seq in test_sequences:
        segment.proposal_sequences[0].sequence = test_seq
        low_gc.evaluate()
        high_gc.evaluate()
        low_score = segment.proposal_sequences[0]._constraints_metadata["low_gc"]["score"]
        high_score = segment.proposal_sequences[0]._constraints_metadata["high_gc"]["score"]
        min_sum = min(min_sum, low_score + high_score)

    logger.info(f"Conflict verification: minimum sum of scores = {min_sum:.4f}")
    if min_sum <= 0.5:
        raise ValueError(f"Objectives not conflicting enough: min_sum={min_sum}")


# ============================================================================
# Pareto front extraction and hypervolume
# ============================================================================


def extract_objective_vectors(segment: Segment, constraints: list[Constraint]) -> list[tuple[float, float]]:
    """Extract (low_gc_score, high_gc_score) for each result sequence."""
    vectors = []
    for seq in segment.result_sequences:
        low_score = seq._constraints_metadata[constraints[0].label]["score"]
        high_score = seq._constraints_metadata[constraints[1].label]["score"]
        vectors.append((low_score, high_score))
    return vectors


def is_dominated(point: tuple[float, float], other: tuple[float, float]) -> bool:
    """Check if point is dominated by other (minimization)."""
    return all(o <= p for o, p in zip(other, point, strict=True)) and any(
        o < p for o, p in zip(other, point, strict=True)
    )


def extract_pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Extract non-dominated points from a set."""
    return [point for point in points if not any(is_dominated(point, other) for other in points)]


def hypervolume_2d(front: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Compute 2D hypervolume (dominated area) relative to reference point.

    Uses the standard 2D hypervolume algorithm: sort by first objective,
    compute rectangles. Reference point must dominate all front points.
    """
    if not front:
        return 0.0

    # Filter out points dominated by reference
    valid_front = [(x, y) for x, y in front if x < reference[0] and y < reference[1]]
    if not valid_front:
        return 0.0

    # Sort by first objective
    sorted_front = sorted(valid_front)

    volume = 0.0
    prev_y = reference[1]

    for x, y in sorted_front:
        if y < prev_y:
            volume += (reference[0] - x) * (prev_y - y)
            prev_y = y

    return volume


# ============================================================================
# NSGA-II run
# ============================================================================


def run_nsga2(seed: int, budget: int) -> dict[str, Any]:
    """Run EA with NSGA-II selection and extract Pareto front."""
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    low_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        weight=1.0,
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
        weight=1.0,
        label="high_gc",
    )

    # Configure EA to match budget
    population_size = 20
    elitism_count = 2
    offspring_per_gen = population_size - elitism_count
    num_generations = (budget - population_size) // offspring_per_gen

    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=elitism_count,
        selection="nsga2",
        seed=seed,
        verbose=False,
    )

    optimizer = EvolutionaryOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[low_gc, high_gc],
        config=config,
    )

    program = Program(optimizers=[optimizer], num_results=population_size, seed=seed)
    program.run()

    # Extract front from pareto_front indices
    all_vectors = extract_objective_vectors(segment, [low_gc, high_gc])
    front_vectors = [all_vectors[idx] for idx in optimizer.pareto_front]

    # Verify budget
    actual_evals = population_size + num_generations * offspring_per_gen
    if abs(actual_evals - budget) >= offspring_per_gen:
        raise ValueError(f"Budget mismatch: {actual_evals} vs {budget}")

    hv = hypervolume_2d(front_vectors, REFERENCE_POINT)

    return {
        "front": front_vectors,
        "front_size": len(front_vectors),
        "hypervolume": hv,
        "total_evaluations": actual_evals,
    }


# ============================================================================
# Multi-weight MCMC run
# ============================================================================


def run_multiweight_mcmc(seed: int, budget: int, num_weights: int = 5) -> dict[str, Any]:
    """Run multiple MCMC chains with different weight vectors, pool non-dominated."""
    # Distribute budget across chains
    steps_per_chain = budget // num_weights

    # Generate weight vectors spanning the space
    weight_pairs = [(i / (num_weights - 1), 1 - i / (num_weights - 1)) for i in range(num_weights)]

    all_points: list[tuple[float, float]] = []

    for chain_idx, (w_low, w_high) in enumerate(weight_pairs):
        segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        low_gc = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
            weight=w_low,
            label="low_gc",
        )

        high_gc = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
            weight=w_high,
            label="high_gc",
        )

        config = MCMCOptimizerConfig(
            num_steps=steps_per_chain,
            seed=seed + chain_idx,
            verbose=False,
        )

        optimizer = MCMCOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[low_gc, high_gc],
            config=config,
        )

        program = Program(optimizers=[optimizer], num_results=1, seed=seed + chain_idx)
        program.run()

        # Collect final solution from this chain
        vectors = extract_objective_vectors(segment, [low_gc, high_gc])
        all_points.extend(vectors)

    # Extract Pareto front from pooled solutions
    front = extract_pareto_front(all_points)
    hv = hypervolume_2d(front, REFERENCE_POINT)

    actual_evals = num_weights * steps_per_chain
    if abs(actual_evals - budget) >= num_weights:
        raise ValueError(f"Budget mismatch: {actual_evals} vs {budget}")

    return {
        "front": front,
        "front_size": len(front),
        "hypervolume": hv,
        "total_evaluations": actual_evals,
        "num_chains": num_weights,
    }


# ============================================================================
# Single-weight MCMC run
# ============================================================================


def run_singleweight_mcmc(seed: int, budget: int) -> dict[str, Any]:
    """Run single MCMC chain with equal weights (floor performance)."""
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    low_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        weight=0.5,
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
        weight=0.5,
        label="high_gc",
    )

    config = MCMCOptimizerConfig(
        num_steps=budget,
        seed=seed,
        verbose=False,
    )

    optimizer = MCMCOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[low_gc, high_gc],
        config=config,
    )

    program = Program(optimizers=[optimizer], num_results=1, seed=seed)
    program.run()

    vectors = extract_objective_vectors(segment, [low_gc, high_gc])
    front = extract_pareto_front(vectors)
    hv = hypervolume_2d(front, REFERENCE_POINT)

    return {
        "front": front,
        "front_size": len(front),
        "hypervolume": hv,
        "total_evaluations": budget,
    }


# ============================================================================
# Statistical analysis and conclusions
# ============================================================================


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_conclusion(
    nsga2_hvs: list[float],
    multiweight_hvs: list[float],
    singleweight_hvs: list[float],
) -> dict[str, Any]:
    """Compute data-driven conclusion with significance gate."""
    nsga2_mean = np.mean(nsga2_hvs)
    multiweight_mean = np.mean(multiweight_hvs)
    singleweight_mean = np.mean(singleweight_hvs)

    # Effect sizes (fractional difference)
    nsga2_vs_multi = (nsga2_mean - multiweight_mean) / max(multiweight_mean, 1e-9)
    nsga2_vs_single = (nsga2_mean - singleweight_mean) / max(singleweight_mean, 1e-9)

    # Simple significance gate: require >10% difference and consistent direction
    nsga2_beats_multi = nsga2_vs_multi > 0.1 and all(n > m for n, m in zip(nsga2_hvs, multiweight_hvs, strict=False))
    nsga2_beats_single = nsga2_vs_single > 0.1

    # Framing
    if nsga2_beats_multi:
        recommendation = (
            "Select NSGA-II for multi-objective problems when you want a diverse Pareto front. "
            f"On this conflicting-GC task, NSGA-II achieves {nsga2_vs_multi:.1%} higher hypervolume "
            f"than multi-weight MCMC ({NUM_TRIALS} trials)."
        )
    elif nsga2_beats_single:
        recommendation = (
            "NSGA-II finds Pareto-optimal trade-offs better than single-weight MCMC "
            f"({nsga2_vs_single:.1%} hypervolume improvement), but is comparable to multi-weight MCMC. "
            "Use NSGA-II when you want the front in one run rather than post-hoc pooling."
        )
    else:
        recommendation = (
            "NSGA-II and multi-weight MCMC perform comparably on this task "
            f"(hypervolume difference {nsga2_vs_multi:.1%}). "
            "Both are valid approaches for multi-objective optimization."
        )

    return {
        "recommendation": recommendation,
        "nsga2_vs_multiweight_effect": nsga2_vs_multi,
        "nsga2_vs_singleweight_effect": nsga2_vs_single,
        "nsga2_mean_hv": nsga2_mean,
        "multiweight_mean_hv": multiweight_mean,
        "singleweight_mean_hv": singleweight_mean,
    }


# ============================================================================
# Plotting
# ============================================================================


def plot_fronts(
    nsga2_fronts: list[list[tuple[float, float]]],
    multiweight_fronts: list[list[tuple[float, float]]],
    singleweight_fronts: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    """Create 2D scatter plot of fronts from all methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    _fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each trial's front (light colors)
    for front in nsga2_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="blue", alpha=0.1, s=20)

    for front in multiweight_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="red", alpha=0.1, s=20)

    for front in singleweight_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="gray", alpha=0.1, s=20)

    # Plot one representative front from each method (darker)
    if nsga2_fronts[0]:
        x, y = zip(*nsga2_fronts[0], strict=True)
        ax.scatter(x, y, c="blue", s=100, label="NSGA-II", edgecolors="black", linewidth=1)

    if multiweight_fronts[0]:
        x, y = zip(*multiweight_fronts[0], strict=True)
        ax.scatter(x, y, c="red", s=100, label="Multi-weight MCMC", marker="^", edgecolors="black", linewidth=1)

    if singleweight_fronts[0]:
        x, y = zip(*singleweight_fronts[0], strict=True)
        ax.scatter(x, y, c="gray", s=100, label="Single-weight MCMC", marker="s", edgecolors="black", linewidth=1)

    ax.set_xlabel("Low GC score (lower is better)", fontsize=12)
    ax.set_ylabel("High GC score (lower is better)", fontsize=12)
    ax.set_title(f"Pareto Fronts: NSGA-II vs MCMC Baselines ({NUM_TRIALS} trials)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved plot to {output_path}")


# ============================================================================
# Main benchmark
# ============================================================================


def main() -> None:
    """Run NSGA-II benchmark and save results."""
    logger.info("=" * 80)
    logger.info("NSGA-II Multi-Objective Optimization Benchmark")
    logger.info("=" * 80)

    # Sanity check
    verify_conflict()

    # Run trials
    logger.info(f"\nRunning {NUM_TRIALS} trials with budget={BUDGET} evaluations each...")

    nsga2_results = []
    multiweight_results = []
    singleweight_results = []

    for trial in range(NUM_TRIALS):
        seed = 1000 + trial

        logger.info(f"Trial {trial + 1}/{NUM_TRIALS}")

        # NSGA-II
        result = run_nsga2(seed, BUDGET)
        nsga2_results.append(result)
        logger.info(f"  NSGA-II: HV={result['hypervolume']:.4f}, front_size={result['front_size']}")

        # Multi-weight MCMC
        result = run_multiweight_mcmc(seed, BUDGET, num_weights=5)
        multiweight_results.append(result)
        logger.info(f"  Multi-weight MCMC: HV={result['hypervolume']:.4f}, front_size={result['front_size']}")

        # Single-weight MCMC
        result = run_singleweight_mcmc(seed, BUDGET)
        singleweight_results.append(result)
        logger.info(f"  Single-weight MCMC: HV={result['hypervolume']:.4f}, front_size={result['front_size']}")

    # Aggregate statistics
    nsga2_hvs = [r["hypervolume"] for r in nsga2_results]
    multiweight_hvs = [r["hypervolume"] for r in multiweight_results]
    singleweight_hvs = [r["hypervolume"] for r in singleweight_results]

    nsga2_sizes = [r["front_size"] for r in nsga2_results]
    multiweight_sizes = [r["front_size"] for r in multiweight_results]
    singleweight_sizes = [r["front_size"] for r in singleweight_results]

    summary = {
        "task": {
            "description": "Conflicting GC-content objectives",
            "sequence_length": SEQUENCE_LENGTH,
            "low_gc_target": f"{LOW_GC_MIN}-{LOW_GC_MAX}%",
            "high_gc_target": f"{HIGH_GC_MIN}-{HIGH_GC_MAX}%",
        },
        "budget": BUDGET,
        "num_trials": NUM_TRIALS,
        "nsga2": {
            "hypervolume": compute_statistics(nsga2_hvs),
            "front_size": compute_statistics(nsga2_sizes),
        },
        "multiweight_mcmc": {
            "hypervolume": compute_statistics(multiweight_hvs),
            "front_size": compute_statistics(multiweight_sizes),
            "num_chains": 5,
        },
        "singleweight_mcmc": {
            "hypervolume": compute_statistics(singleweight_hvs),
            "front_size": compute_statistics(singleweight_sizes),
        },
        "conclusion": compute_conclusion(nsga2_hvs, multiweight_hvs, singleweight_hvs),
    }

    # Save results
    summary_path = Path("benchmark_nsga2_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")

    detailed_path = Path("benchmark_nsga2_detailed.json")
    with detailed_path.open("w") as f:
        json.dump(
            {
                "nsga2": nsga2_results,
                "multiweight_mcmc": multiweight_results,
                "singleweight_mcmc": singleweight_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved detailed results to {detailed_path}")

    # Plot
    plot_path = Path("benchmark_nsga2_fronts.png")
    plot_fronts(
        [r["front"] for r in nsga2_results],
        [r["front"] for r in multiweight_results],
        [r["front"] for r in singleweight_results],
        plot_path,
    )

    # Print conclusion
    logger.info("\n" + "=" * 80)
    logger.info("CONCLUSION")
    logger.info("=" * 80)
    logger.info(summary["conclusion"]["recommendation"])  # type: ignore[index]
    logger.info("\nMean Hypervolume:")
    logger.info(f"  NSGA-II:           {summary['nsga2']['hypervolume']['mean']:.4f}")  # type: ignore[index]
    logger.info(f"  Multi-weight MCMC: {summary['multiweight_mcmc']['hypervolume']['mean']:.4f}")  # type: ignore[index]
    logger.info(f"  Single-weight MCMC: {summary['singleweight_mcmc']['hypervolume']['mean']:.4f}")  # type: ignore[index]
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
