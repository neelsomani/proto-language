"""Tests for the ESM2 perplexity constraint."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proto_language import (
    ESM2PerplexityConfig,
    esm2_perplexity_constraint,
    esm2_perplexity_gradient_backward,
)
from proto_language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.core import Sequence
from proto_language.utils import one_hot_protein_matrix
from proto_language.utils.sequence_matrices import SequenceLogitBiasConfig

_TOOL_MODULE = "proto_language.constraint.sequence_scoring.esm2_perplexity_constraint"


def _seq_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


def _mock_gradient_output(
    *, gradient: list[list[float]] | None, loss: float = 0.5, log_likelihood: float = -2.5
) -> SimpleNamespace:
    """Build a mock ESM2GradientOutput-shaped object."""
    return SimpleNamespace(
        gradient=gradient,
        loss=loss,
        metrics={
            "log_likelihood": log_likelihood,
            "avg_log_likelihood": -loss,
            "perplexity": float(np.exp(loss)),
        },
    )


class TestBackward:
    @patch(f"{_TOOL_MODULE}.run_esm2_gradient")
    def test_backward_passes_full_binder_logits(self, mock_run: object) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.1] * 20] * 5, loss=0.5)
        binder = _seq_with_logits(np.ones((5, 20)) / 20.0)

        (result,) = esm2_perplexity_gradient_backward(
            [(binder,)],
            config=ESM2PerplexityConfig(temperature=0.8, use_ste=False, device="cpu"),
        )

        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.temperature == 0.8
        assert len(tool_input.logits) == 5 and len(tool_input.logits[0]) == 20
        assert tool_config.use_ste is False and tool_config.compute_gradient is True
        assert tool_config.model_checkpoint == "esm2_t33_650M_UR50D"
        assert len(result.gradient) == 1 and result.gradient[0].shape == (5, 20)

    @patch(f"{_TOOL_MODULE}.run_esm2_gradient")
    def test_backward_applies_logit_scale_and_sequence_bias(self, mock_run: object) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 2, loss=1.0)
        binder = _seq_with_logits(np.ones((2, 20)))

        (result,) = esm2_perplexity_gradient_backward(
            [(binder,)],
            config=ESM2PerplexityConfig(
                temperature=0.6,
                device="cpu",
                logit_scale=2.0,
                sequence_bias=SequenceLogitBiasConfig(reference_sequence="AA", reference_bias=0.25),
            ),
        )

        # logit_scale=2.0 doubles raw_logits=1.0 → 2.0; sequence_bias adds +0.25 at the A column.
        tool_input = mock_run.call_args[0][0]
        assert tool_input.logits[0][0] == 2.25
        assert result.gradient[0][0, 0] == pytest.approx(1.0)

    @patch(f"{_TOOL_MODULE}.run_esm2_gradient")
    def test_ppl_backward_scales_nll_gradient_by_perplexity(self, mock_run: object) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 5, loss=2.0)
        binder = _seq_with_logits(np.zeros((5, 20)))

        (result,) = esm2_perplexity_gradient_backward(
            [(binder,)], config=ESM2PerplexityConfig(temperature=0.6, device="cpu", score_mode="ppl")
        )

        assert result.loss == pytest.approx(np.exp(2.0))
        assert result.gradient[0][0, 0] == pytest.approx(0.5 * np.exp(2.0))
        assert result.metrics["esm2_nll"] == 2.0
        assert result.metrics["esm2_score_mode"] == "ppl"

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @patch(f"{_TOOL_MODULE}.run_esm2_gradient")
    def test_ppl_backward_rejects_overflow(self, mock_run: object) -> None:
        """Guards the chain-rule scaling step from leaking ``inf`` when ``exp(NLL)`` overflows."""
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 3, loss=1000.0)
        binder = _seq_with_logits(np.zeros((3, 20)))

        with pytest.raises(ValueError, match="non-finite"):
            esm2_perplexity_gradient_backward(
                [(binder,)], config=ESM2PerplexityConfig(temperature=0.6, device="cpu", score_mode="ppl")
            )


class TestForward:
    @pytest.mark.parametrize(("score_mode", "expected_score"), [("nll", 2.0), ("ppl", pytest.approx(np.exp(2.0)))])
    def test_returns_requested_score_and_metadata(self, score_mode: str, expected_score: object) -> None:
        binder = Sequence("MKTAY", "protein")

        with patch(f"{_TOOL_MODULE}.run_esm2_gradient") as mock_run:
            mock_run.return_value = _mock_gradient_output(gradient=None, loss=2.0, log_likelihood=-10.0)
            (result,) = esm2_perplexity_constraint(
                [(binder,)], config=ESM2PerplexityConfig(temperature=0.6, device="cpu", score_mode=score_mode)
            )

        assert result.score == expected_score
        assert result.metadata["esm2_log_likelihood"] == -10.0
        assert result.metadata["esm2_avg_log_likelihood"] == -2.0
        assert result.metadata["esm2_loss"] == 2.0
        assert result.metadata["esm2_nll"] == 2.0
        assert result.metadata["esm2_perplexity"] == pytest.approx(np.exp(2.0))
        assert result.metadata["esm2_score_mode"] == score_mode
        assert result.metadata["esm2_model_checkpoint"] == "esm2_t33_650M_UR50D"
        assert mock_run.call_args[0][1].compute_gradient is False


class TestRegistry:
    """Constraint registers as dual-mode with one binder slot."""

    def test_registers_as_dual_mode_with_one_slot(self) -> None:
        spec = ConstraintRegistry.get("esm2-perplexity")
        assert spec.mode == "dual"
        assert spec.function is esm2_perplexity_constraint
        assert spec.backward is esm2_perplexity_gradient_backward
        assert spec.input_labels == [InputSlot(label="Sequence", requires_logits=True)]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_real_model_forward_backward_consistency() -> None:
    """End-to-end check against real ESM2 on a 14-residue GFP N-terminal fragment.

    Pins three properties in one model load:
      1. Forward (discrete) and backward (one-hot logits) share the same NLL objective.
      2. Forward metadata is internally consistent (perplexity == exp(NLL)) and reports
         the configured checkpoint.
      3. Backward returns a finite, non-zero `(L, 20)` gradient.

    Uses ``esm2_t6_8M_UR50D`` (smallest checkpoint) to keep runtime modest.
    """
    seq = "MSKGEELFTGVVPI"  # GFP N-terminus
    checkpoint = "esm2_t6_8M_UR50D"
    config = ESM2PerplexityConfig(model_checkpoint=checkpoint, temperature=1.0, use_ste=True)

    (forward,) = esm2_perplexity_constraint([(Sequence(seq, "protein"),)], config=config)

    one_hot = Sequence(seq, "protein")
    one_hot.logits = np.array(one_hot_protein_matrix(seq))
    (backward,) = esm2_perplexity_gradient_backward([(one_hot,)], config=config)

    assert forward.score == pytest.approx(backward.loss, rel=1e-5)

    assert forward.metadata["esm2_perplexity"] == pytest.approx(math.exp(forward.score), rel=1e-5)
    assert forward.metadata["esm2_model_checkpoint"] == checkpoint

    grad = backward.gradient[0]
    assert grad.shape == (len(seq), 20)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.0)
