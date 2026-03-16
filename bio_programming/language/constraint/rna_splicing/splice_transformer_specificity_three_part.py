"""
Evaluate tissue-specific splicing with SpliceTransformer using three-part input.

This variant accepts three segments (left_flank, intron_core, right_flank) and
concatenates them into a single target sequence before scoring. This enables
multicontext optimization where the intron segment is shared across constructs
with different flanking sequences.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.constraint.rna_splicing.splice_transformer_specificity import (
    SpliceTransformerSpecificityConfig,
    splice_transformer_specificity,
)
from proto_language.language.core import Sequence

logger = logging.getLogger(__name__)


@constraint(
    key="splice-transformer-specificity-three-part",
    label="SpliceTransformer tissue specificity score (three-part)",
    config=SpliceTransformerSpecificityConfig,
    description=(
        "Evaluate tissue-specific splicing with SpliceTransformer. "
        "Takes three segments (left_flank, intron_core, right_flank), "
        "concatenates them into the 1-kb target, and scores tissue specificity."
    ),
    uses_gpu=True,
    tools_called=["splice-transformer-prediction"],
    category="rna splicing",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=3,
)
def splice_transformer_specificity_three_part(
    input_sequences: List[Tuple[Sequence, ...]],
    config: SpliceTransformerSpecificityConfig,
) -> List[float]:
    """Evaluate tissue-specific splicing using three-part concatenated input.

    Each input tuple contains (left_flank, intron_core, right_flank) which are
    concatenated into a single target sequence before passing to the base
    SpliceTransformer specificity scorer. Metadata from the concatenated
    evaluation is propagated back to all three input segments.

    Args:
        input_sequences: List of 3-tuples (left_flank, intron_core, right_flank).
        config: Same config as the single-segment variant.

    Returns:
        List of float scores in [0.0, 1.0]. Interpretation depends on direction.
    """
    concatenated_inputs = []
    for left_flank, intron_core, right_flank in input_sequences:
        concatenated = Sequence(
            sequence=left_flank.sequence + intron_core.sequence + right_flank.sequence,
            sequence_type=left_flank.sequence_type,
        )
        concatenated_inputs.append((concatenated,))

    scores = splice_transformer_specificity(concatenated_inputs, config)

    # Propagate metadata from concatenated sequence back to all three parts.
    for (left_flank, intron_core, right_flank), (concatenated,) in zip(
        input_sequences, concatenated_inputs
    ):
        metadata_update = dict(concatenated._metadata)
        for seq in (left_flank, intron_core, right_flank):
            seq._metadata.update(metadata_update)

    return scores
