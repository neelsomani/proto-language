"""
ESMFold pTM constraint for protein structure quality evaluation.
"""

from __future__ import annotations

from typing import Optional

from ...base import Sequence, SequenceType
from ....schemas import ESMFoldKwargs
from ..utils import run_esmfold


def esmfold_ptm_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted TM-score (pTM).

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of sequence replications (default: 1).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score where 0.0 indicates perfect structure quality (pTM = 1.0)
        and higher values indicate lower structure quality.

    Examples:
        Evaluating protein structure quality:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = esmfold_ptm_constraint(seq, 1, kwargs)
    """

    run_esmfold(input_sequence, n_replications, esmfold_kwargs)
    return 1.0 - input_sequence._metadata["ptm"]