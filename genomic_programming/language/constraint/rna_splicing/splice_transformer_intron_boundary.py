"""
Evaluate intron boundary prediction with SpliceTransformer.
"""
import numpy as np
from pydantic import Field
from typing import List, Optional

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.models.rna_splicing.splice_transformer import (
    run_splice_transformer,
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerType,
    TARGET_LENGTH as SPLICE_TRANSFORMER_TARGET_LENGTH,
    CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH,
)


class SpliceTransformerIntronBoundaryConfig(BaseConfig):
    """Configuration for SpliceTransformer constraint."""
    left_context: str = Field(
        description="Sequence of the left context",
    )
    right_context: str = Field(
        description="Sequence of the right context",
    )
    donor_pos: int | List[int] = Field(
        description="0-indexed position(s) into input_sequence of expected donor",
    )
    acceptor_pos: int | List[int] = Field(
        description="0-indexed position(s) into input_sequence of expected acceptor",
    )
    splice_transformer_config: Optional[SpliceTransformerConfig] = Field(
        default=None,
        description="Optional parameter configuration for SpliceTransformer. If None, default values are used.",
    )


@ConstraintRegistry.register(
    key="splice-transformer-intron-boundary",
    label="SpliceTransformer intron boundary score",
    config=SpliceTransformerIntronBoundaryConfig,
    description="Evaluate intron boundary prediction with SpliceTransformer",
    vectorized=False,
    concatenate=True,
    gpu_required=True
)
def splice_transformer_intron_boundary(
    input_sequence: Sequence,
    config: SpliceTransformerIntronBoundaryConfig,
) -> float:
    """
    Evaluate intron boundary prediction with SpliceTransformer.

    The input_sequence itself must be a sequence of length 1000. The config should provide the 4-kb
    left and right contexts that are also required for SpliceTransformer.

    SpliceTransformer is trained to make the actual prediction for the donor on the position
    right before the "GT" and for the acceptor on the position right after the "AG."
    """
    assert len(config.left_context) == len(config.right_context) == SPLICE_TRANSFORMER_CONTEXT_LENGTH, \
        f"Context lengths must be {SPLICE_TRANFORMER_CONTEXT_LENGTH}"
    context_length = len(config.left_context)

    donor_pos = [config.donor_pos] \
        if isinstance(config.donor_pos, int) else config.donor_pos
    acceptor_pos = [config.acceptor_pos] \
        if isinstance(config.acceptor_pos, int) else config.acceptor_pos

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

    donor_score = float(output[:, donor_pos, SpliceTransformerType.DONOR.value].mean())
    acceptor_score = float(output[:, acceptor_pos, SpliceTransformerType.ACCEPTOR.value].mean())
    score = 1. - ((donor_score + acceptor_score) / 2)

    input_sequence._metadata.update({
        "donor_pos": config.donor_pos,
        "acceptor_pos": config.acceptor_pos,
        "donor_score": 1. - donor_score,
        "acceptor_score": 1. - acceptor_score,
        "total_splice_score": score,
    })

    return score
