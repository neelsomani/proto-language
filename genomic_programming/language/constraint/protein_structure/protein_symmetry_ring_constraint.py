"""
Protein symmetry ring constraint for symmetric multimeric structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np

from ...base import Sequence, SequenceType
from ....schemas import ESMFoldKwargs
from ....utils import (
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
)
from ..utils import run_esmfold


def protein_symmetry_ring_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    all_to_all_protomer_symmetry: bool = False,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Constrain a protein to form a symmetric ring-like multimeric structure.

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of protomers in the ring (default: 1).
        all_to_all_protomer_symmetry: Use all pairwise distances vs adjacent (default: False).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.

    Examples:
        Evaluating ring symmetry:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = protein_symmetry_ring_constraint(seq, 6, False, kwargs)  # Hexameric ring
    """
    from biotite.structure import get_chains

    run_esmfold(input_sequence, n_replications, esmfold_kwargs)

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

    centroids = []
    for chain_id in get_chains(atom_array):
        chain_backbone = get_backbone_atoms(
            atom_array[atom_array.chain_id == chain_id]
        ).coord
        centroids.append(get_centroid(chain_backbone))

    assert len(centroids) == n_replications
    centroids = np.vstack(centroids)

    distance_func = (
        pairwise_distances if all_to_all_protomer_symmetry else adjacent_distances
    )
    return float(np.std(distance_func(centroids)))