"""
Overall protein quality constraint function.
"""

from __future__ import annotations

from typing import Any, Dict

from ...base import Sequence, SequenceType
from ....tools.orf_prediction.prodigal import run_prodigal
from ..utils import validate_required_config
from .protein_length_constraint import protein_length_constraint
from .protein_complexity_constraint import protein_complexity_constraint
from .protein_repetitiveness_constraint import protein_repetitiveness_constraint
from .protein_diversity_constraint import protein_diversity_constraint
from .balanced_aa_constraint import balanced_aa_constraint


def overall_protein_quality_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate protein quality either from predicted proteins (DNA input) or directly (protein input).

    For DNA sequences, runs Prodigal first to predict proteins, then checks all predicted
    proteins. For protein sequences, checks the sequence directly.

    Args:
        input_sequence: The DNA or protein sequence to analyze.
        config: Configuration dictionary containing:
            For DNA input:
                - min_high_quality_fraction (float): Minimum fraction of predicted proteins that must be high quality.
                - protein_quality_config (dict): Configuration dictionary with the following structure:
                {
                    "min_high_quality_fraction": 0.8,  # Minimum fraction of predicted proteins that must be high quality (0.0-1.0)
                    "protein_quality_config": {
                        "quality_threshold": 0.1,  # Maximum acceptable constraint score for a protein to be considered "high quality"

                        # Individual protein quality constraints (all optional):
                        "length": {
                            "min_length": 50,     # Minimum acceptable protein length (amino acids)
                            "max_length": 2000    # Maximum acceptable protein length (amino acids)
                        },
                        "complexity": {
                            "max_low_complexity": 0.3,              # Maximum fraction of low-complexity regions (0.0-1.0)
                            "segmasker_path": "segmasker"           # Path to segmasker executable (optional)
                        },
                        "repetitiveness": {
                            "max_repetitiveness": 0.4,              # Maximum acceptable repetitiveness fraction (0.0-1.0)
                            "min_repeat_length": 3                  # Minimum repeat length to consider (optional, default: 3)
                        },
                        "diversity": {
                            "min_diversity": 0.3                    # Minimum acceptable amino acid diversity (0.0-1.0, where 1.0 = all 20 amino acids)
                        },
                        "balanced_aas": {
                            "max_underrepresented": 0.2             # Maximum acceptable fraction of underrepresented amino acids (0.0-1.0)
                        }
                    }
                }

            For protein input:
                - protein_quality_config (dict): Configuration for protein quality checks with the following structure:
                {
                    "protein_quality_config": {
                        "quality_threshold": 0.1,  # Maximum acceptable constraint score for overall quality assessment

                        # Same individual constraints as above (all optional)
                        "length": { ... },
                        "complexity": { ... },
                        "repetitiveness": { ... },
                        "diversity": { ... },
                        "balanced_aas": { ... }
                    }
                }

    Returns:
        Constraint score between 0.0 and 1.0 where:
        - 0.0 indicates perfect/optimal protein quality (all constraints satisfied)
        - Values closer to 0.0 indicate better constraint satisfaction
        - 1.0 indicates worst possible protein quality (maximum constraint violation)

    Example:
        >>> dna_seq = Sequence("ATGAAACGTATTGCGTCG...", SequenceType.DNA)
        >>> minimal_config = {
        ...     "min_high_quality_fraction": 0.5,
        ...     "protein_quality_config": {
        ...         "quality_threshold": 0.2,
        ...         "length": {
        ...             "min_length": 100,
        ...             "max_length": 800
        ...         }
        ...     }
        ... }
        >>> score = overall_protein_quality_constraint(dna_seq, minimal_config)
    """
    validate_required_config(config, ["protein_quality_config"])
    protein_config = config["protein_quality_config"]

    if input_sequence.sequence_type == SequenceType.DNA:
        # For DNA sequences: predict proteins first
        validate_required_config(config, ["min_high_quality_fraction"])
        min_high_quality_fraction = config["min_high_quality_fraction"]

        # Get predicted proteins, this will load cached proteins if they already exist
        proteins_df = run_prodigal(input_sequence)

        if len(proteins_df) == 0:
            input_sequence._metadata["predicted_protein_count"] = 0
            input_sequence._metadata["high_quality_protein_count"] = 0
            input_sequence._metadata["high_quality_protein_fraction"] = 0.0
            input_sequence._metadata["protein_quality_details"] = []
            return 1.0  # Maximum penalty for no proteins found

        # Evaluate each predicted protein
        high_quality_count = 0
        protein_quality_details = []
        all_protein_avg_scores = []

        for idx, protein_row in proteins_df.iterrows():
            protein_seq = Sequence(protein_row["sequence"], SequenceType.PROTEIN)

            # Apply all protein quality constraints
            quality_scores = {}
            overall_scores = []

            if "length" in protein_config:
                score = protein_length_constraint(protein_seq, protein_config["length"])
                quality_scores["length"] = score
                overall_scores.append(quality_scores["length"])

            if "complexity" in protein_config:
                score = protein_complexity_constraint(
                    protein_seq, protein_config["complexity"]
                )
                quality_scores["complexity"] = score
                overall_scores.append(quality_scores["complexity"])

            if "repetitiveness" in protein_config:
                score = protein_repetitiveness_constraint(
                    protein_seq, protein_config["repetitiveness"]
                )
                quality_scores["repetitiveness"] = score
                overall_scores.append(quality_scores["repetitiveness"])

            if "diversity" in protein_config:
                score = protein_diversity_constraint(
                    protein_seq, protein_config["diversity"]
                )
                quality_scores["diversity"] = score
                overall_scores.append(quality_scores["diversity"])

            if "balanced_aas" in protein_config:
                score = balanced_aa_constraint(
                    protein_seq, protein_config["balanced_aas"]
                )
                quality_scores["balanced_aas"] = score
                overall_scores.append(quality_scores["balanced_aas"])

            # Calculate average score for this protein
            avg_score = (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            )
            all_protein_avg_scores.append(avg_score)

            # Consider protein high quality if average score is below threshold
            threshold = protein_config.get("quality_threshold", 0.1)
            is_high_quality = avg_score <= threshold

            if is_high_quality:
                high_quality_count += 1

            protein_quality_details.append(
                {
                    "protein_id": protein_row["id"],
                    "length": len(protein_row["sequence"]),
                    "is_high_quality": is_high_quality,
                    "avg_constraint_score": avg_score,
                    "quality_scores": quality_scores,
                    "metadata": protein_seq._metadata.copy(),
                }
            )

        high_quality_fraction = high_quality_count / len(proteins_df)

        # Store comprehensive metadata
        input_sequence._metadata["predicted_protein_count"] = len(proteins_df)
        input_sequence._metadata["high_quality_protein_count"] = high_quality_count
        input_sequence._metadata["high_quality_protein_fraction"] = (
            high_quality_fraction
        )
        input_sequence._metadata["protein_quality_details"] = protein_quality_details
        input_sequence._metadata["protein_quality_threshold"] = protein_config.get(
            "quality_threshold", 0.1
        )

        # If we high quality fraction requirement is met, return 0
        if high_quality_fraction >= min_high_quality_fraction:
            return 0.0

        # Otherwise, return a score based on how far we are from meeting the requirement
        overall_avg_protein_score = sum(all_protein_avg_scores) / len(
            all_protein_avg_scores
        )
        fraction_deficit = (
            min_high_quality_fraction - high_quality_fraction
        ) / min_high_quality_fraction

        # Combine the average protein quality with the fraction deficit
        combined_score = (overall_avg_protein_score + fraction_deficit) / 2.0
        return min(1.0, max(0.0, combined_score))

    elif input_sequence.sequence_type == SequenceType.PROTEIN:
        # For protein sequences: evaluate quality directly on input sequence
        quality_scores = {}
        overall_scores = []

        if "length" in protein_config:
            score = protein_length_constraint(input_sequence, protein_config["length"])
            quality_scores["length"] = score
            overall_scores.append(quality_scores["length"])

        if "complexity" in protein_config:
            score = protein_complexity_constraint(
                input_sequence, protein_config["complexity"]
            )
            quality_scores["complexity"] = score
            overall_scores.append(quality_scores["complexity"])

        if "repetitiveness" in protein_config:
            score = protein_repetitiveness_constraint(
                input_sequence, protein_config["repetitiveness"]
            )
            quality_scores["repetitiveness"] = score
            overall_scores.append(quality_scores["repetitiveness"])

        if "diversity" in protein_config:
            score = protein_diversity_constraint(
                input_sequence, protein_config["diversity"]
            )
            quality_scores["diversity"] = score
            overall_scores.append(quality_scores["diversity"])

        if "balanced_aas" in protein_config:
            score = balanced_aa_constraint(
                input_sequence, protein_config["balanced_aas"]
            )
            quality_scores["balanced_aas"] = score
            overall_scores.append(quality_scores["balanced_aas"])

        # Calculate overall quality score as average of individual constraint scores
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        threshold = protein_config.get("quality_threshold", 0.0)
        is_high_quality = avg_score <= threshold

        # Store metadata for protein input
        input_sequence._metadata["protein_quality_scores"] = quality_scores
        input_sequence._metadata["avg_constraint_score"] = avg_score
        input_sequence._metadata["is_high_quality"] = is_high_quality
        input_sequence._metadata["protein_quality_threshold"] = threshold

        # If protein meets quality threshold, return 0, otherwise return the average score
        if is_high_quality:
            return 0.0
        else:
            return min(1.0, max(0.0, avg_score))

    else:
        raise ValueError("Input sequence must be either DNA or PROTEIN type")
