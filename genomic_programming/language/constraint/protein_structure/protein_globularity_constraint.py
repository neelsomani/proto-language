"""
Protein globularity constraint for compact protein structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional,List

import numpy as np
from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.models.structure_prediction import (
    run_esmfold,
    StructurePredictionComplex,
    ESMFoldInput,
    ESMFoldConfig,
)
from proto_language.utils import (
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
    MAX_ENERGY
)
from proto_language.tools.models.structure_prediction.esmfold import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
)
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)


MAX_GLOBULARITY = 20.0

class ProteinGlobularityConfig(BaseConfig):
    """Configuration for protein globularity constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters. Leave as None to use defaults.",
    )


@ConstraintRegistry.register(
    key="protein-globularity",
    label="Protein Globularity",
    config=ProteinGlobularityConfig,
    description="Encourage compact, globular protein structures",
    vectorized=True,
    concatenate=True,
    gpu_required=True
)
def protein_globularity_constraint(sequences: List[Sequence], config: ProteinGlobularityConfig) -> List[float]:
    """
    Encourage compact, globular protein structures.

    Supports both protein and DNA sequences:
    - Protein: Direct structure prediction
    - DNA: Uses Prodigal to predict proteins first, then evaluates their structures

    Args:
        sequences: List of protein or DNA sequences to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.

    Returns:
        List of constraint scores based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.
    """

    scores = []
    if sequences[0].sequence_type == SequenceType.PROTEIN:
        scores = _evaluate_protein_globularity(sequences, config)
    else:
        scores = _evaluate_dna_globularity(sequences, config)

    return scores


def _evaluate_protein_globularity(
    protein_sequences: List[Sequence], config: ProteinGlobularityConfig
) -> List[float]:
    """Evaluate protein globularity directly."""

    # Create complexes with n_replications of each protein sequence
    complexes = [
        StructurePredictionComplex(
            chains=[seq.sequence] * config.n_replications,
            entity_types=["protein"] * config.n_replications,
        )
        for seq in protein_sequences
    ]

    # Create the ESMFold input containing all the complexes
    esmfold_input = ESMFoldInput(complexes=complexes)

    # Run ESMFold
    output = run_esmfold(
        inputs=esmfold_input, config=config.esmfold_config or ESMFoldConfig()
    )

    scores = []
    for protein_seq, comp, structure in zip(
        protein_sequences, complexes, output.structures
    ):
        protein_seq._metadata.update(
            {
                "avg_plddt": structure.avg_plddt,
                "ptm": structure.ptm,
                "pdb_output": structure.structure_pdb_output,
                "esmfolded_sequence": comp.chains,
            }
        )

        # Calculate globularity from structure
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb_output))
        backbone = get_backbone_atoms(atom_array).coord
        globularity_score = float(np.std(distances_to_centroid(backbone)))

        # Update the globularity score in the metadata
        protein_seq._metadata["globularity_score"] = globularity_score
        scores.append(globularity_score)

    return scores


def _evaluate_dna_globularity(
    dna_sequences: List[Sequence],
    config: ProteinGlobularityConfig
) -> List[float]:
    """Evaluate DNA sequences via Prodigal then globularity."""
    prodigal_result = run_prodigal_prediction(
        ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences]),
        ProdigalConfig()
    )

    scores = []

    for dna_seq, proteins_df, num_genes in zip(
        dna_sequences,
        prodigal_result.results_per_sequence,
        prodigal_result.total_num_genes_per_sequence
    ):
        dna_seq._metadata.update({
            "prodigal_proteins": proteins_df,
            "prodigal_protein_count": num_genes
        })

        if num_genes == 0 or len(proteins_df) == 0:
            scores.append(MAX_ENERGY)
            continue

        protein_seqs = proteins_df['protein_sequence'].tolist()
        complexes = [
            StructurePredictionComplex(
                chains=[seq] * config.n_replications,
                entity_types=["protein"] * config.n_replications,
            )
            for seq in protein_seqs
        ]

        # Run ESMFold
        esmfold_output = run_esmfold(
            inputs=ESMFoldInput(complexes=complexes),
            config=config.esmfold_config or ESMFoldConfig(),
        )

        # Calculate globularity for all proteins, use best (lowest std)
        globularities = []
        for structure in esmfold_output.structures:
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb_output))
            backbone = get_backbone_atoms(atom_array).coord
            globularities.append(float(np.std(distances_to_centroid(backbone))))

        best_globularity = min(globularities)
        globularity_score = min(1.0, best_globularity / MAX_GLOBULARITY)
        dna_seq._metadata["esmfold_protein_globularities"] = globularities
        dna_seq._metadata["esmfold_best_globularity"] = best_globularity
        dna_seq._metadata["esmfold_normalized_globularity"] = globularity_score
        scores.append(globularity_score)

    return scores
