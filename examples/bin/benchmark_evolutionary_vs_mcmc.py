"""Benchmark comparing EvolutionaryOptimizer vs MCMCOptimizer.

This benchmark compares the two optimizers on a controlled, deterministic task
(GC-content optimization) with matched constraint-evaluation budgets. The goal
is to measure:
1. Best valid candidate found per N constraint evaluations
2. Number of distinct valid candidates found
3. Convergence curves over the optimization run

The EA's advantage is diversity: it maintains a population that explores multiple
promising regions simultaneously, while MCMC follows a single trajectory that can
get stuck in one basin.

Outputs (written to current directory):
    - benchmark_ea_vs_mcmc_summary.json: Aggregate statistics across trials
    - benchmark_ea_vs_mcmc_detailed.json: Per-trial results and sequences
    - benchmark_ea_vs_mcmc_convergence.png: Convergence plot (first trial)

Usage:
    python examples/bin/benchmark_evolutionary_vs_mcmc.py
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_evolutionary_optimizer(
    sequence_length: int,
    target_gc: float,
    population_size: int,
    num_generations: int,
    seed: int,
) -> dict[str, Any]:
    """Run evolutionary optimizer and collect metrics.

    Args:
        sequence_length (int): Length of sequences to optimize.
        target_gc (float): Target GC content percentage (0-100).
        population_size (int): Population size for EA.
        num_generations (int): Number of generations to run.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Metrics including best_score, distinct_candidates, convergence, etc.
    """
    # Set up segment and generator
    segment = Segment(sequence="A" * sequence_length, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # GC-content constraint (deterministic, fast)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=target_gc, max_gc=target_gc),
    )

    # Configure EA
    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=max(1, population_size // 10),  # 10% elitism
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=seed,
        verbose=False,
        tracking_interval=1,  # Track every generation
    )

    optimizer = EvolutionaryOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[constraint],
        config=config,
    )

    # Run optimization
    program = Program(optimizers=[optimizer], num_results=population_size, seed=seed)
    program.run()

    # Extract metrics from history
    convergence = []
    all_sequences = set()
    valid_sequences = set()
    total_evaluations = 0

    for snapshot in optimizer.history:
        time_step = snapshot.get("time_step", 0)
        results = snapshot.get("results", [])

        # Count evaluations (population_size per generation, except generation 0)
        if time_step > 0:
            total_evaluations += population_size

        # Extract energy scores and sequences from results
        energy_scores = []
        for result in results:
            energy_score = result.get("energy_score")
            energy_scores.append(energy_score)

            # Collect sequences from constructs
            for construct in result.get("constructs", []):
                for segment_data in construct.get("segments", []):
                    seq = segment_data.get("sequence", "")
                    if seq:
                        all_sequences.add(seq)
                        # Valid if energy score is finite and reasonably good
                        if energy_score is not None and math.isfinite(energy_score):
                            valid_sequences.add(seq)

        # Best score this generation
        finite_scores = [s for s in energy_scores if s is not None and math.isfinite(s)]
        if finite_scores:
            best_score = min(finite_scores)
            mean_score = float(np.mean(finite_scores))
            convergence.append(
                {
                    "generation": time_step,
                    "evaluations": total_evaluations,
                    "best_score": best_score,
                    "mean_score": mean_score,
                    "distinct_sequences": len(all_sequences),
                    "distinct_valid": len(valid_sequences),
                }
            )

    # Final metrics
    final_scores = [s for s in optimizer.energy_scores if math.isfinite(s)]
    best_score = min(final_scores) if final_scores else float("inf")

    return {
        "optimizer": "evolutionary",
        "best_score": best_score,
        "distinct_candidates": len(all_sequences),
        "distinct_valid_candidates": len(valid_sequences),
        "total_evaluations": total_evaluations,
        "convergence": convergence,
        "final_sequences": [seq.sequence for seq in segment.result_sequences],
    }


def run_mcmc_optimizer(
    sequence_length: int,
    target_gc: float,
    num_results: int,
    proposals_per_result: int,
    num_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run MCMC optimizer and collect metrics.

    Args:
        sequence_length (int): Length of sequences to optimize.
        target_gc (float): Target GC content percentage (0-100).
        num_results (int): Number of independent MCMC trajectories.
        proposals_per_result (int): Proposals per trajectory per step.
        num_steps (int): Number of MCMC steps to run.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Metrics including best_score, distinct_candidates, convergence, etc.
    """
    # Set up segment and generator
    segment = Segment(sequence="A" * sequence_length, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # GC-content constraint (deterministic, fast)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=target_gc, max_gc=target_gc),
    )

    # Configure MCMC
    config = MCMCOptimizerConfig(
        num_results=num_results,
        proposals_per_result=proposals_per_result,
        num_steps=num_steps,
        seed=seed,
        verbose=False,
        tracking_interval=1,  # Track every step
    )

    optimizer = MCMCOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[constraint],
        config=config,
    )

    # Run optimization
    program = Program(optimizers=[optimizer], num_results=num_results, seed=seed)
    program.run()

    # Extract metrics from history
    convergence = []
    all_sequences = set()
    valid_sequences = set()
    total_evaluations = 0

    for snapshot in optimizer.history:
        time_step = snapshot.get("time_step", 0)
        results = snapshot.get("results", [])

        # Count evaluations (num_results * proposals_per_result per step, except step 0)
        if time_step > 0:
            total_evaluations += num_results * proposals_per_result

        # Extract energy scores and sequences from results
        energy_scores = []
        for result in results:
            energy_score = result.get("energy_score")
            energy_scores.append(energy_score)

            # Collect sequences from constructs
            for construct in result.get("constructs", []):
                for segment_data in construct.get("segments", []):
                    seq = segment_data.get("sequence", "")
                    if seq:
                        all_sequences.add(seq)
                        if energy_score is not None and math.isfinite(energy_score):
                            valid_sequences.add(seq)

        # Best score this step
        finite_scores = [s for s in energy_scores if s is not None and math.isfinite(s)]
        if finite_scores:
            best_score = min(finite_scores)
            mean_score = float(np.mean(finite_scores))
            convergence.append(
                {
                    "step": time_step,
                    "evaluations": total_evaluations,
                    "best_score": best_score,
                    "mean_score": mean_score,
                    "distinct_sequences": len(all_sequences),
                    "distinct_valid": len(valid_sequences),
                }
            )

    # Final metrics
    final_scores = [s for s in optimizer.energy_scores if math.isfinite(s)]
    best_score = min(final_scores) if final_scores else float("inf")

    return {
        "optimizer": "mcmc",
        "best_score": best_score,
        "distinct_candidates": len(all_sequences),
        "distinct_valid_candidates": len(valid_sequences),
        "total_evaluations": total_evaluations,
        "convergence": convergence,
        "final_sequences": [seq.sequence for seq in segment.result_sequences],
    }


def plot_convergence(ea_results: dict[str, Any], mcmc_results: dict[str, Any], output_path: Path) -> None:
    """Plot convergence curves comparing EA and MCMC.

    Args:
        ea_results (dict): Results from EA optimizer.
        mcmc_results (dict): Results from MCMC optimizer.
        output_path (Path): Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evolutionary Algorithm vs MCMC Comparison", fontsize=16)

    # Extract convergence data
    ea_conv = ea_results["convergence"]
    mcmc_conv = mcmc_results["convergence"]

    ea_evals = [c["evaluations"] for c in ea_conv]
    ea_best = [c["best_score"] for c in ea_conv]
    ea_mean = [c["mean_score"] for c in ea_conv]
    ea_distinct = [c["distinct_sequences"] for c in ea_conv]
    ea_valid = [c["distinct_valid"] for c in ea_conv]

    mcmc_evals = [c["evaluations"] for c in mcmc_conv]
    mcmc_best = [c["best_score"] for c in mcmc_conv]
    mcmc_mean = [c["mean_score"] for c in mcmc_conv]
    mcmc_distinct = [c["distinct_sequences"] for c in mcmc_conv]
    mcmc_valid = [c["distinct_valid"] for c in mcmc_conv]

    # Plot 1: Best score vs evaluations
    axes[0, 0].plot(ea_evals, ea_best, "b-", label="EA", linewidth=2)
    axes[0, 0].plot(mcmc_evals, mcmc_best, "r--", label="MCMC", linewidth=2)
    axes[0, 0].set_xlabel("Constraint Evaluations")
    axes[0, 0].set_ylabel("Best Score (lower is better)")
    axes[0, 0].set_title("Convergence: Best Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean score vs evaluations
    axes[0, 1].plot(ea_evals, ea_mean, "b-", label="EA", linewidth=2)
    axes[0, 1].plot(mcmc_evals, mcmc_mean, "r--", label="MCMC", linewidth=2)
    axes[0, 1].set_xlabel("Constraint Evaluations")
    axes[0, 1].set_ylabel("Mean Score")
    axes[0, 1].set_title("Convergence: Mean Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Distinct sequences explored
    axes[1, 0].plot(ea_evals, ea_distinct, "b-", label="EA", linewidth=2)
    axes[1, 0].plot(mcmc_evals, mcmc_distinct, "r--", label="MCMC", linewidth=2)
    axes[1, 0].set_xlabel("Constraint Evaluations")
    axes[1, 0].set_ylabel("Distinct Sequences Explored")
    axes[1, 0].set_title("Diversity: Total Distinct Sequences")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Distinct valid sequences
    axes[1, 1].plot(ea_evals, ea_valid, "b-", label="EA", linewidth=2)
    axes[1, 1].plot(mcmc_evals, mcmc_valid, "r--", label="MCMC", linewidth=2)
    axes[1, 1].set_xlabel("Constraint Evaluations")
    axes[1, 1].set_ylabel("Distinct Valid Sequences")
    axes[1, 1].set_title("Diversity: Distinct Valid Candidates")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Convergence plot saved to {output_path}")


def run_benchmark(
    budget: int,
    sequence_length: int = 50,
    target_gc: float = 50.0,
    num_trials: int = 5,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    """Run benchmark comparing EA vs MCMC with matched evaluation budgets.

    Args:
        budget (int): Constraint evaluation budget (total evaluations).
        sequence_length (int): Length of sequences to optimize.
        target_gc (float): Target GC content percentage.
        num_trials (int): Number of independent trials to run.
        seed (int): Base random seed.
        output_dir (Path | None): Directory to save results.
    """
    if output_dir is None:
        output_dir = Path(".")  # Current directory

    logger.info(f"Starting benchmark: budget={budget}, seq_len={sequence_length}, target_gc={target_gc}%")
    logger.info(f"Running {num_trials} trials with base seed {seed}")

    all_ea_results = []
    all_mcmc_results = []

    for trial in range(num_trials):
        trial_seed = seed + trial
        logger.info(f"\n=== Trial {trial + 1}/{num_trials} (seed={trial_seed}) ===")

        # EA configuration: population evolves over generations
        # Budget = population_size * num_generations
        population_size = 20  # Fixed population size
        num_generations = budget // population_size

        logger.info(f"EA: population_size={population_size}, num_generations={num_generations}")
        ea_results = run_evolutionary_optimizer(
            sequence_length=sequence_length,
            target_gc=target_gc,
            population_size=population_size,
            num_generations=num_generations,
            seed=trial_seed,
        )
        all_ea_results.append(ea_results)
        logger.info(
            f"EA Results: best_score={ea_results['best_score']:.6f}, "
            f"distinct_valid={ea_results['distinct_valid_candidates']}"
        )

        # MCMC configuration: multiple trajectories with proposals per step
        # Budget = num_results * proposals_per_result * num_steps
        num_results = 20  # Match EA population size
        proposals_per_result = 1  # Standard MCMC (one proposal per trajectory)
        num_steps = budget // (num_results * proposals_per_result)

        logger.info(f"MCMC: num_results={num_results}, proposals_per_result={proposals_per_result}, num_steps={num_steps}")
        mcmc_results = run_mcmc_optimizer(
            sequence_length=sequence_length,
            target_gc=target_gc,
            num_results=num_results,
            proposals_per_result=proposals_per_result,
            num_steps=num_steps,
            seed=trial_seed,
        )
        all_mcmc_results.append(mcmc_results)
        logger.info(
            f"MCMC Results: best_score={mcmc_results['best_score']:.6f}, "
            f"distinct_valid={mcmc_results['distinct_valid_candidates']}"
        )

    # Aggregate results across trials
    ea_best_scores = [r["best_score"] for r in all_ea_results]
    ea_distinct_valid = [r["distinct_valid_candidates"] for r in all_ea_results]
    mcmc_best_scores = [r["best_score"] for r in all_mcmc_results]
    mcmc_distinct_valid = [r["distinct_valid_candidates"] for r in all_mcmc_results]

    summary = {
        "budget": budget,
        "sequence_length": sequence_length,
        "target_gc": target_gc,
        "num_trials": num_trials,
        "seed": seed,
        "ea": {
            "best_score_mean": float(np.mean(ea_best_scores)),
            "best_score_std": float(np.std(ea_best_scores)),
            "best_score_min": float(np.min(ea_best_scores)),
            "distinct_valid_mean": float(np.mean(ea_distinct_valid)),
            "distinct_valid_std": float(np.std(ea_distinct_valid)),
        },
        "mcmc": {
            "best_score_mean": float(np.mean(mcmc_best_scores)),
            "best_score_std": float(np.std(mcmc_best_scores)),
            "best_score_min": float(np.min(mcmc_best_scores)),
            "distinct_valid_mean": float(np.mean(mcmc_distinct_valid)),
            "distinct_valid_std": float(np.std(mcmc_distinct_valid)),
        },
    }

    # Save summary
    summary_path = output_dir / "benchmark_ea_vs_mcmc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Budget: {budget} constraint evaluations")
    logger.info(f"Trials: {num_trials}")
    logger.info("")
    logger.info("Best Score (lower is better):")
    logger.info(f"  EA:   {summary['ea']['best_score_mean']:.6f} ± {summary['ea']['best_score_std']:.6f}")
    logger.info(f"  MCMC: {summary['mcmc']['best_score_mean']:.6f} ± {summary['mcmc']['best_score_std']:.6f}")
    logger.info("")
    logger.info("Distinct Valid Candidates:")
    logger.info(f"  EA:   {summary['ea']['distinct_valid_mean']:.1f} ± {summary['ea']['distinct_valid_std']:.1f}")
    logger.info(f"  MCMC: {summary['mcmc']['distinct_valid_mean']:.1f} ± {summary['mcmc']['distinct_valid_std']:.1f}")
    logger.info("=" * 60)

    # Plot convergence for first trial
    if all_ea_results and all_mcmc_results:
        plot_path = output_dir / "benchmark_ea_vs_mcmc_convergence.png"
        plot_convergence(all_ea_results[0], all_mcmc_results[0], plot_path)

    # Save detailed results
    detailed_path = output_dir / "benchmark_ea_vs_mcmc_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(
            {
                "ea_results": all_ea_results,
                "mcmc_results": all_mcmc_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Detailed results saved to {detailed_path}")


def main() -> None:
    """Run benchmark comparing EvolutionaryOptimizer vs MCMCOptimizer."""
    setup_logging(verbose=False)

    # Hardcoded benchmark parameters (like benchmark_human_protein_generators.py)
    budget = 1000
    sequence_length = 50
    target_gc = 50.0
    num_trials = 5
    seed = 42

    run_benchmark(
        budget=budget,
        sequence_length=sequence_length,
        target_gc=target_gc,
        num_trials=num_trials,
        seed=seed,
        output_dir=None,  # Write to current directory
    )


if __name__ == "__main__":
    main()
