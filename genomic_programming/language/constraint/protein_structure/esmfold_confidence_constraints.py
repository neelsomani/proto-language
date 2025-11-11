"""
esmfold_confidence_constraints.py

Contains implementation of ESMFold structure quality constraints.
- ESMFold pLDDT constraint
- ESMFold pTM constraint
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.tools.structure_prediction import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
    StructurePredictionComplex,
)
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.utils import MAX_ENERGY


class ESMFoldConfidenceConfig(BaseConfig):
    """Configuration for ESMFold pLDDT constraint."""

    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers, 2+ for oligomers (dimers, trimers, etc.). Higher values increase computational cost.",
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Optional ESMFold configuration (residue_idx_offset, chain_linker, verbose). If None, uses defaults.",
    )


@ConstraintRegistry.register(
    key="esmfold-plddt",
    label="ESMFold pLDDT Score",
    config=ESMFoldConfidenceConfig,
    description="Evaluate protein structure quality using ESMFold predicted LDDT score",
    batched=True,
    concatenate=True,
    gpu_required=True,
)
def esmfold_plddt_constraint(
    sequences: List[Sequence], config: ESMFoldConfidenceConfig
) -> List[float]:
    """
    Evaluate protein structure quality using ESMFold's predicted LDDT (pLDDT) score.
    """
    return esmfold_confidence(sequences, config, target_metric="avg_plddt")


@ConstraintRegistry.register(
    key="esmfold-ptm",
    label="ESMFold pTM Score",
    config=ESMFoldConfidenceConfig,
    description="Evaluate protein structure quality using ESMFold predicted TM score",
    batched=True,
    concatenate=True,
    gpu_required=True,
)
def esmfold_ptm_constraint(
    sequences: List[Sequence], config: ESMFoldConfidenceConfig
) -> List[float]:
    """
    Evaluate protein structure quality using ESMFold's predicted TM (pTM) score.
    """
    return esmfold_confidence(sequences, config, target_metric="ptm")


def esmfold_confidence(
    sequences: List[Sequence],
    config: ESMFoldConfidenceConfig,
    target_metric: str = "avg_plddt",
) -> List[float]:
    """
    Helper function to evaluate protein structure confidence using ESMFold.

    Args:
        sequences: The protein sequences to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.
        target_metric: The metric to evaluate. Either "plddt" or "ptm".

    Returns:
        Constraint score where 0.0 indicates perfect structure confidence (pLDDT = 1.0)
        and higher values indicate lower structure confidence.

    Examples:
        Evaluating protein structure confidence:

        >>> seqs = [Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)]
        >>> # Using defaults:
        >>> cfg = ESMFoldPLDDTConfig()
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
        >>> # With custom ESMFold parameters:
        >>> esmfold_cfg = ESMFoldConfig(residue_idx_offset=256, verbose=True)
        >>> cfg = ESMFoldPLDDTConfig(n_replications=2, esmfold_config=esmfold_cfg)
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
    """
    if target_metric not in ["avg_plddt", "ptm"]:
        raise ValueError(
            f"Invalid target metric: {target_metric}. Must be 'avg_plddt' or 'ptm'."
        )

    scores = []
    if sequences[0].sequence_type == SequenceType.PROTEIN:
        scores = _predict_structures(sequences, config, target_metric)
    else:
        scores = _evaluate_dna_via_prodigal(sequences, config, target_metric)

    return scores


def _predict_structures(
    sequences: List[Sequence], config: ESMFoldConfidenceConfig, target_metric: str
) -> List[float]:
    """Predict structures and return specified confidence scores."""

    # Create complexes with n_replications of each protein sequence
    complexes = [
        StructurePredictionComplex(
            chains=[seq.sequence] * config.n_replications,
            entity_types=["protein"] * config.n_replications,
        )
        for seq in sequences
    ]

    esmfold_input = ESMFoldInput(complexes=complexes)
    output = run_esmfold(
        inputs=esmfold_input, config=config.esmfold_config or ESMFoldConfig()
    )
    scores = []
    for seq, comp, structure in zip(sequences, complexes, output):
        seq._metadata.update(
            {
                "avg_plddt": structure.avg_plddt,
                "ptm": structure.ptm,
                "pdb_output": structure.structure_pdb_output,
                "esmfolded_sequence": ":".join(comp.chains),
            }
        )

        # Calculate constraint score (lower pLDDT = higher penalty)
        scores.append(1.0 - getattr(structure, target_metric))

    return scores


def _evaluate_dna_via_prodigal(
    dna_sequences: List[Sequence], config: ESMFoldConfidenceConfig, target_metric: str
) -> List[float]:
    """Evaluate DNA sequences by predicting proteins with Prodigal first."""
    # Batch predict all DNA sequences at once
    prodigal_result = run_prodigal_prediction(
        ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences]),
        ProdigalConfig(),
    )

    scores = []

    # Process each DNA sequence's results
    for i, (dna_seq, proteins_df, num_genes) in enumerate(
        zip(
            dna_sequences,
            prodigal_result.results_per_sequence,
            prodigal_result.total_num_genes_per_sequence,
        )
    ):
        # Store Prodigal results
        dna_seq._metadata.update(
            {"prodigal_proteins": proteins_df, "prodigal_protein_count": num_genes}
        )

        if num_genes == 0:
            scores.append(MAX_ENERGY)
            continue

        if len(proteins_df) == 0:
            scores.append(MAX_ENERGY)
            continue

        # Predict structures for this DNA's proteins
        protein_seqs = proteins_df["protein_sequence"].tolist()
        complexes = [
            StructurePredictionComplex(
                chains=[seq] * config.n_replications,
                entity_types=["protein"] * config.n_replications,
            )
            for seq in protein_seqs
        ]

        esmfold_output = run_esmfold(
            inputs=ESMFoldInput(complexes=complexes),
            config=config.esmfold_config or ESMFoldConfig(),
        )

        plddts = [structure.avg_plddt for structure in esmfold_output]
        ptms = [structure.ptm for structure in esmfold_output]

        dna_seq._metadata["esmfold_protein_plddts"] = plddts
        dna_seq._metadata["esmfold_protein_ptms"] = ptms

        best_plddt = max(plddts)
        best_ptm = max(ptms)

        dna_seq._metadata["esmfold_best_plddt"] = best_plddt
        dna_seq._metadata["esmfold_best_ptm"] = best_ptm

        if target_metric == "avg_plddt":
            score = 1.0 - best_plddt
        else:
            score = 1.0 - best_ptm

        scores.append(score)

    return scores
