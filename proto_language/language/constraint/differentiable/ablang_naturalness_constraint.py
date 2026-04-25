"""AbLang antibody naturalness constraint (dual-mode: discrete scoring + gradient).

Takes one binder ``Segment``. Optional ``heavy_slice`` / ``light_slice`` config
fields enable single-chain scFv mode (paired VH+VL call); unset scores the whole
binder as a heavy-only chain (VHH / nanobody mode).
"""

from typing import Any

import numpy as np
from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang import (
    AbLangGradientConfig,
    AbLangGradientInput,
    run_ablang_gradient,
)
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.language.core.constraint import GradientConstraintOutput
from proto_language.utils import one_hot_protein_matrix


class AbLangConstraintConfig(BaseConfig):
    """Configuration for AbLang naturalness scoring (forward + gradient).

    Attributes:
        temperature (float): Softmax temperature for AbLang. Required, no default:
            AbLang's fixed temperature is a scientific parameter of the pipeline (Germinal
            VHH uses 0.6, ``vhh.yaml:46``), not a framework default, so callers must
            choose it deliberately.
        use_ste (bool): Use Straight-Through Estimator (hard one-hot forward pass with
            gradients through soft probabilities). Germinal always uses STE.
        device (str): Execution device for AbLang, for example ``"cuda"`` or ``"cpu"``.
        heavy_slice (tuple[int, int] | None): Optional half-open ``(start, end)`` over the
            binder Segment for the VH region. Set together with ``light_slice`` to enable
            single-chain scFv mode; leave both ``None`` for VHH (heavy-only) scoring.
        light_slice (tuple[int, int] | None): Optional half-open ``(start, end)`` over the
            binder Segment for the VL region. Set together with ``heavy_slice``.
        logit_scale (float): Multiply raw logits before AbLang. 2.0 for Germinal parity.
        logit_bias (list[list[float]] | None): Additive bias (L x 20) applied after scaling.
    """

    temperature: float = ConfigField(
        title="AbLang Temperature",
        gt=0.0,
        description="Softmax temperature for AbLang (fixed, not varied per step like AF2).",
    )
    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=True,
        description="Hard one-hot forward pass with soft-probability gradients. Matches Germinal's default.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Execution device for AbLang, for example 'cuda' or 'cpu'.",
        hidden=True,
    )
    heavy_slice: tuple[int, int] | None = ConfigField(
        title="Heavy Chain Slice",
        default=None,
        description="VH region (start, end) within the binder; set with light_slice for scFv mode.",
    )
    light_slice: tuple[int, int] | None = ConfigField(
        title="Light Chain Slice",
        default=None,
        description="VL region (start, end) within the binder; set with heavy_slice for scFv mode.",
    )
    logit_scale: float = ConfigField(
        title="Logit Scale",
        default=1.0,
        gt=0.0,
        description="Pre-scale raw logits before AbLang. Set to 2.0 for Germinal parity.",
    )
    logit_bias: list[list[float]] | None = ConfigField(
        title="Logit Bias",
        default=None,
        description="Additive bias (L x 20) applied after scaling, before AbLang.",
        hidden=True,
    )

    @model_validator(mode="after")
    def _validate_slices(self) -> "AbLangConstraintConfig":
        """Slices must be both set or both None; each non-empty; non-overlapping."""
        heavy, light = self.heavy_slice, self.light_slice
        if (heavy is None) != (light is None):
            raise ValueError("heavy_slice and light_slice must be set together (both None for VHH mode).")
        if heavy is None or light is None:
            return self
        for name, (start, end) in (("heavy_slice", heavy), ("light_slice", light)):
            if start < 0 or end <= start:
                raise ValueError(f"{name}={(start, end)} must be a non-empty range with start >= 0 and end > start.")
        if heavy[0] < light[1] and light[0] < heavy[1]:
            raise ValueError(f"heavy_slice {heavy} overlaps light_slice {light}.")
        return self


def ablang_naturalness_gradient_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangConstraintConfig,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Compute AbLang naturalness gradient w.r.t. binder logits (batched).

    VHH mode (no slices): the whole binder is one heavy chain. scFv mode (slices set):
    slice VH and VL out of the binder, call AbLang paired, scatter per-chain gradients
    back into a full-binder-shaped array with linker rows zero.
    """
    results: list[GradientConstraintOutput] = []
    for (binder_seq,) in input_sequences:
        raw_logits = binder_seq.logits
        assert raw_logits is not None  # noqa: S101 -- input_labels slot check guarantees it
        bias = np.asarray(config.logit_bias, dtype=np.float64) if config.logit_bias is not None else 0.0
        logits = config.logit_scale * raw_logits + bias

        if config.heavy_slice is None:
            output = run_ablang_gradient(
                AbLangGradientInput(
                    antibody=AntibodyLogits(heavy_chain=logits.tolist()), temperature=config.temperature
                ),
                AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
            )
            assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
            grad = np.array(output.gradient, dtype=np.float64) * config.logit_scale
            results.append(GradientConstraintOutput(gradient=(grad,), loss=output.loss, metrics=output.metrics))
            continue

        assert config.light_slice is not None  # noqa: S101 -- validator guarantees both-or-neither
        h_start, h_end = config.heavy_slice
        l_start, l_end = config.light_slice
        if max(h_end, l_end) > logits.shape[0]:
            raise ValueError(
                f"slices (heavy={config.heavy_slice}, light={config.light_slice}) extend past binder length {logits.shape[0]}."
            )
        vh_logits, vl_logits = logits[h_start:h_end], logits[l_start:l_end]
        output = run_ablang_gradient(
            AbLangGradientInput(
                antibody=AntibodyLogits(heavy_chain=vh_logits.tolist(), light_chain=vl_logits.tolist()),
                temperature=config.temperature,
            ),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
        )
        assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
        paired_grad = np.array(output.gradient, dtype=np.float64) * config.logit_scale
        full_grad = np.zeros_like(raw_logits, dtype=np.float64)
        full_grad[h_start:h_end] = paired_grad[: h_end - h_start]
        full_grad[l_start:l_end] = paired_grad[h_end - h_start :]
        results.append(GradientConstraintOutput(gradient=(full_grad,), loss=output.loss, metrics=output.metrics))
    return results


@constraint(
    key="ablang-naturalness",
    label="AbLang Naturalness",
    config=AbLangConstraintConfig,
    description="AbLang naturalness on a single binder Segment (VHH/nanobody by default; set heavy_slice/light_slice to score a single-chain scFv as paired VH+VL). Discrete scoring or gradient w.r.t. logits.",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="Binder", requires_logits=True)],
    backward=ablang_naturalness_gradient_backward,
    backward_config=AbLangConstraintConfig,
)
def ablang_naturalness_forward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangConstraintConfig,
) -> list[ConstraintOutput]:
    """Forward AbLang naturalness scoring via masked pseudo-log-likelihood.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal ``(binder_seq,)``.
        config (AbLangConstraintConfig): Forward-mode config; slice fields control mode.

    Returns:
        list[ConstraintOutput]: Per-proposal raw AbLang loss (lower is better) with
            ``ablang_log_likelihood`` and ``ablang_loss`` metadata.
    """
    results: list[ConstraintOutput] = []
    for (binder_seq,) in input_sequences:
        if config.heavy_slice is None:
            antibody = AntibodyLogits(heavy_chain=one_hot_protein_matrix(binder_seq.sequence))
        else:
            assert config.light_slice is not None  # noqa: S101 -- validator guarantees both-or-neither
            h_start, h_end = config.heavy_slice
            l_start, l_end = config.light_slice
            if max(h_end, l_end) > len(binder_seq.sequence):
                raise ValueError(
                    f"slices (heavy={config.heavy_slice}, light={config.light_slice}) extend past binder length {len(binder_seq.sequence)}."
                )
            antibody = AntibodyLogits(
                heavy_chain=one_hot_protein_matrix(binder_seq.sequence[h_start:h_end]),
                light_chain=one_hot_protein_matrix(binder_seq.sequence[l_start:l_end]),
            )
        output = run_ablang_gradient(
            AbLangGradientInput(antibody=antibody, temperature=config.temperature),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=False, device=config.device),
        )
        results.append(
            ConstraintOutput(
                score=output.loss,
                metadata={
                    "ablang_log_likelihood": output.metrics["log_likelihood"],
                    "ablang_loss": output.loss,
                },
            )
        )
    return results


# Germinal semigreedy ranks proposals on the raw antibody-LM loss, so this
# intentional exception keeps discrete Stage-2 scoring aligned with that target.
ablang_naturalness_forward._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
