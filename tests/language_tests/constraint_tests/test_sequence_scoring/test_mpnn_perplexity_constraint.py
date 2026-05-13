"""Tests for the ProteinMPNN perplexity constraint."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proto_language.language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.language.constraint.sequence_scoring.mpnn_perplexity_constraint import (
    MpnnPerplexityConfig,
    mpnn_perplexity_constraint,
    mpnn_perplexity_gradient_backward,
)
from proto_language.language.core import Sequence
from proto_language.utils import one_hot_protein_matrix
from proto_language.utils.sequence_logit_bias import SequenceLogitBiasConfig

_TOOL_MODULE = "proto_language.language.constraint.sequence_scoring.mpnn_perplexity_constraint"


def _seq_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


def _mock_gradient_output(*, gradient: list[list[float]] | None, loss: float = 0.5) -> SimpleNamespace:
    """Build a mock ProteinMPNNGradientOutput-shaped object."""
    return SimpleNamespace(
        gradient=gradient,
        loss=loss,
        metrics={
            "log_likelihood": -loss * 5.0,
            "avg_log_likelihood": -loss,
            "perplexity": float(np.exp(loss)),
            "effective_sequence_length": 5.0,
        },
    )


class TestForward:
    """Forward scoring calls ProteinMPNN gradient tool in forward-only mode."""

    @pytest.mark.parametrize(
        ("score_mode", "expected_score"),
        [("nll", 1.5), ("ppl", pytest.approx(np.exp(1.5)))],
    )
    def test_returns_requested_score_and_metadata(
        self, score_mode: str, expected_score: object, sample_pdb_content: str
    ) -> None:
        with patch(f"{_TOOL_MODULE}.run_proteinmpnn_gradient") as mock_run:
            mock_run.return_value = _mock_gradient_output(gradient=None, loss=1.5)
            (result,) = mpnn_perplexity_constraint(
                [(Sequence("AGSVL", "protein"),)],
                config=MpnnPerplexityConfig(
                    structure_input={"structure": sample_pdb_content, "chains_to_redesign": ["A"]},
                    device="cpu",
                    score_mode=score_mode,
                ),
            )

        assert result.score == expected_score
        assert result.metadata["mpnn_nll"] == 1.5
        assert result.metadata["mpnn_perplexity"] == pytest.approx(np.exp(1.5))
        assert result.metadata["mpnn_score_mode"] == score_mode
        assert result.metadata["mpnn_model_choice"] == "proteinmpnn"
        assert mock_run.call_args[0][1].compute_gradient is False


class TestBackward:
    """Backward (gradient) mode propagates config to the tool and applies the ppl chain rule."""

    @patch(f"{_TOOL_MODULE}.run_proteinmpnn_gradient")
    def test_propagates_structure_temperature_and_compute_gradient(
        self, mock_run: object, sample_pdb_content: str
    ) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.1] * 20] * 5, loss=0.5)
        binder = _seq_with_logits(np.ones((5, 20)) / 20.0)

        (result,) = mpnn_perplexity_gradient_backward(
            [(binder,)],
            config=MpnnPerplexityConfig(
                structure_input={
                    "structure": sample_pdb_content,
                    "chains_to_redesign": ["A"],
                    "fixed_positions": {"A": [1]},
                },
                temperature=0.7,
                device="cpu",
            ),
        )

        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.temperature == 0.7
        assert tool_input.chains_to_redesign.chains == ["A"]
        assert tool_input.fixed_positions.chains == {"A": [1]}
        assert tool_config.compute_gradient is True
        assert result.gradient[0].shape == (5, 20)

    @patch(f"{_TOOL_MODULE}.run_proteinmpnn_gradient")
    def test_applies_logit_scale_and_sequence_bias(self, mock_run: object, sample_pdb_content: str) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 2, loss=1.0)
        binder = _seq_with_logits(np.ones((2, 20)))

        (result,) = mpnn_perplexity_gradient_backward(
            [(binder,)],
            config=MpnnPerplexityConfig(
                structure_input={"structure": sample_pdb_content, "chains_to_redesign": ["A"]},
                device="cpu",
                logit_scale=2.0,
                sequence_bias=SequenceLogitBiasConfig(reference_sequence="AA", reference_bias=0.25),
                score_mode="nll",
            ),
        )

        # logit_scale=2.0 doubles raw_logits=1.0 → 2.0; sequence_bias adds +0.25 at the A column.
        # NLL-mode gradient is scaled by logit_scale: 0.5 * 2.0 = 1.0.
        tool_input = mock_run.call_args[0][0]
        assert tool_input.logits[0][0] == 2.25
        assert result.gradient[0][0, 0] == pytest.approx(1.0)

    @patch(f"{_TOOL_MODULE}.run_proteinmpnn_gradient")
    def test_ppl_mode_scales_nll_gradient_by_perplexity(self, mock_run: object, sample_pdb_content: str) -> None:
        """Chain rule: d(exp(NLL))/dx = exp(NLL) * d(NLL)/dx."""
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 5, loss=2.0)
        binder = _seq_with_logits(np.zeros((5, 20)))

        (result,) = mpnn_perplexity_gradient_backward(
            [(binder,)],
            config=MpnnPerplexityConfig(
                structure_input={"structure": sample_pdb_content, "chains_to_redesign": ["A"]},
                device="cpu",
                score_mode="ppl",
            ),
        )

        assert result.loss == pytest.approx(np.exp(2.0))
        assert result.gradient[0][0, 0] == pytest.approx(0.5 * np.exp(2.0))
        assert result.metrics["mpnn_nll"] == 2.0
        assert result.metrics["mpnn_score_mode"] == "ppl"

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @patch(f"{_TOOL_MODULE}.run_proteinmpnn_gradient")
    def test_ppl_mode_rejects_overflow(self, mock_run: object, sample_pdb_content: str) -> None:
        """Guards the ppl chain-rule from leaking ``inf`` when ``exp(NLL)`` overflows."""
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 3, loss=1000.0)
        binder = _seq_with_logits(np.zeros((3, 20)))

        with pytest.raises(ValueError, match="non-finite"):
            mpnn_perplexity_gradient_backward(
                [(binder,)],
                config=MpnnPerplexityConfig(
                    structure_input={"structure": sample_pdb_content, "chains_to_redesign": ["A"]},
                    device="cpu",
                    score_mode="ppl",
                ),
            )


class TestRegistry:
    def test_registers_as_dual_mode_with_one_logits_slot(self) -> None:
        spec = ConstraintRegistry.get("mpnn-perplexity")
        assert spec.mode == "dual"
        assert spec.function is mpnn_perplexity_constraint
        assert spec.backward is mpnn_perplexity_gradient_backward
        assert spec.input_labels == [InputSlot(label="Sequence", requires_logits=True)]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_real_model_forward_backward_consistency(sample_pdb_content: str) -> None:
    """End-to-end check against real ProteinMPNN on the 5-residue test backbone.

    Pins three properties in one model load:
      1. Forward (discrete) and backward (one-hot logits) share the same NLL objective.
      2. Forward metadata is internally consistent (perplexity == exp(NLL)).
      3. Backward returns a finite, non-zero ``(L, 20)`` gradient.
    """
    seq = "AGSVL"
    config = MpnnPerplexityConfig(
        structure_input={"structure": sample_pdb_content, "chains_to_redesign": ["A"]},
        score_mode="nll",
        seed=0,
    )

    (forward,) = mpnn_perplexity_constraint([(Sequence(seq, "protein"),)], config=config)

    one_hot = Sequence(seq, "protein")
    one_hot.logits = np.array(one_hot_protein_matrix(seq))
    (backward,) = mpnn_perplexity_gradient_backward([(one_hot,)], config=config)

    assert forward.score == pytest.approx(backward.loss, rel=1e-4)
    assert forward.metadata["mpnn_perplexity"] == pytest.approx(math.exp(forward.score), rel=1e-5)

    grad = backward.gradient[0]
    assert grad.shape == (len(seq), 20)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.0)
