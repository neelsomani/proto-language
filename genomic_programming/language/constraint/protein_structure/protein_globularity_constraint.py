"""
Protein globularity constraint for compact protein structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np

from ...base import Sequence, SequenceType
from ....schemas import ESMFoldKwargs
from ....utils import (
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
)
from ..utils import run_esmfold


def protein_globularity_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Encourage compact, globular protein structures.

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of sequence replications (default: 1).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.

    Examples:
        Evaluating protein globularity:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = protein_globularity_constraint(seq, 1, kwargs)
    """
    run_esmfold(input_sequence, n_replications, esmfold_kwargs)

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))
    backbone = get_backbone_atoms(atom_array).coord
    return float(np.std(distances_to_centroid(backbone)))