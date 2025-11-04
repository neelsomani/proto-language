"""
Evaluate tissue-specific splicing with SpliceTransformer.
"""
from enum import Enum
from pydantic import Field
from typing import List, Optional

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.models.rna_splicing.splice_transformer import (
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerTissue,
    TISSUE_INDEX_OFFSET,
    run_splice_transformer,
)


class SpliceTransformerSpecificityConfig(BaseConfig):
    """Configuration for SpliceTransformer constraint."""
    left_context: str = Field(
        description="Sequence of the left context",
    )
    right_context: str = Field(
        description="Sequence of the right context",
    )
    splice_pos: int | List[int] = Field(
        description="0-indexed position(s) into input_sequence on which to compute the score",
    )
    tissue: str = Field(
        default="AVERAGE",
        description="Tissue on which to define the score (must be in SpliceTransformerTissue). Defaults to AVERAGE, which averages across all tissues."
    )
    direction: str = Field(
        default="max",
        description="Whether to maximize or minimize the value. Defaults to 'max' (other value is 'min')."
    )
    splice_transformer_config: Optional[SpliceTransformerConfig] = Field(
        default=None,
        description="Optional parameter configuration for SpliceTransformer. If None, default values are used.",
    )


@ConstraintRegistry.register(
    key="splice-transformer-specificity",
    label="SpliceTransformer tissue specificity score",
    config=SpliceTransformerSpecificityConfig,
    description="Evaluate tissue specific splicing with SpliceTransformer",
    vectorized=False,
    concatenate=True,
    gpu_required=True
)
def splice_transformer_specificity(
    input_sequence: Sequence,
    config: SpliceTransformerSpecificityConfig,
) -> float:
    """
    Evaluate a potential sequence for tissue specific splicing with SpliceTransformer.

    The input_sequence itself must be a sequence of length 1000. The config should provide the 4-kb
    left and right contexts that are also required for SpliceTransformer.

    SpliceTransformer is trained to make the actual prediction for the donor on the position
    right before the "GT" and for the acceptor on the position right after the "AG."
    """
    assert len(config.left_context) == len(config.right_context)
    context_length = len(config.left_context)
    splice_pos = [config.splice_pos] \
        if isinstance(config.splice_pos, int) else config.splice_pos
    tissue = SpliceTransformerTissue[config.tissue]

    splice_transformer_input = SpliceTransformerInput(
        target_seqs=[input_sequence.sequence],
        left_contexts=[config.left_context],
        right_contexts=[config.right_context],
    )
    if config.splice_transformer_config is None:
        splice_transformer_config = SpliceTransformerConfig(context_length=context_length)
    else:
        splice_transformer_config = config.splice_transformer_config

    output = run_splice_transformer(
        splice_transformer_input,
        splice_transformer_config,
    ).prediction

    assert output.shape[1] == len(input_sequence.sequence)

    if tissue == SpliceTransformerTissue.AVERAGE:
        score = float(output[:, splice_pos, TISSUE_INDEX_OFFSET:].mean())
    else:
        score = float(output[:, splice_pos, TISSUE_INDEX_OFFSET + tissue.value].mean())

    if config.direction == "max":
        score = 1. - score
    elif config.direction == "min":
        pass
    else:
        raise ValueError(
            f"Invalid SpliceTransformer specificity direction: {config.direction}, "
            "must be either 'max' or 'min'."
        )

    input_sequence._metadata.update({
        f"specificity_direction_{config.tissue}": config.direction,
        f"specificity_score_{config.tissue}": score,
    })

    return score
