"""Gradient math utilities: mergers, norm alignment, normalization, and Adam/SGD steps."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

GradientMergerName = Literal["weighted_sum", "pcgrad", "mgda"]


class GradientMerger(ABC):
    """Abstract base class for gradient merging strategies."""

    @abstractmethod
    def merge(self, gradients: list[np.ndarray], weights: list[float] | np.ndarray | None = None) -> np.ndarray:
        """Merge multiple gradients into a single update direction.

        ``weights`` is for standalone use; ``GradientOptimizer`` pre-weights the
        gradients (so PCGrad/MGDA project on weighted vectors) and passes ``None``.
        """

    @staticmethod
    def _prepare_inputs(
        gradients: list[np.ndarray], weights: list[float] | np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize merger inputs."""
        if not gradients:
            raise ValueError("gradients must contain at least one array")

        try:
            stacked = np.stack([np.asarray(gradient, dtype=float) for gradient in gradients])
        except ValueError as exc:
            raise ValueError("all gradients must have the same shape") from exc

        if not np.isfinite(stacked).all():
            raise ValueError("all gradients must contain only finite values")

        normalized_weights = (
            np.ones(stacked.shape[0], dtype=float) if weights is None else np.asarray(weights, dtype=float)
        )
        if normalized_weights.shape != (stacked.shape[0],):
            raise ValueError("weights must have the same length as gradients")
        if not np.isfinite(normalized_weights).all():
            raise ValueError("weights must contain only finite values")

        return stacked, normalized_weights

    @classmethod
    def _flatten_inputs(
        cls, gradients: list[np.ndarray], weights: list[float] | np.ndarray | None
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """Return weighted, flattened gradients plus the original gradient shape.

        Weights are applied before flattening so conflict resolution (PCGrad, MGDA)
        operates on weighted vectors — a deliberate weight-then-project design.
        """
        stacked, normalized_weights = cls._prepare_inputs(gradients, weights)
        return stacked.shape[1:], stacked.reshape(stacked.shape[0], -1) * normalized_weights[:, None]


class WeightedSumMerger(GradientMerger):
    """Simple weighted-sum merger."""

    def merge(self, gradients: list[np.ndarray], weights: list[float] | np.ndarray | None = None) -> np.ndarray:
        """Merge gradients via weighted summation."""
        stacked, normalized_weights = self._prepare_inputs(gradients, weights)
        return np.asarray(np.tensordot(normalized_weights, stacked, axes=1), dtype=float)


class PCGradMerger(GradientMerger):
    """Pairwise-conflict merger using deterministic projection order.

    Each gradient is projected against every other **original** gradient (not
    against already-projected ones). This differs from Yu et al. 2020, which
    shuffles task order and projects iteratively against the running result.
    Both variants are reasonable; this one is deterministic across runs.
    """

    def merge(self, gradients: list[np.ndarray], weights: list[float] | np.ndarray | None = None) -> np.ndarray:
        """Merge gradients via pairwise conflict projection."""
        shape, flattened = self._flatten_inputs(gradients, weights)
        projected = []

        for index, gradient in enumerate(flattened):
            current = gradient.copy()
            for other_index, other in enumerate(flattened):
                if index == other_index:
                    continue
                dot_product = float(np.dot(current, other))
                other_norm = float(np.dot(other, other))
                if dot_product < 0 and other_norm > 0:
                    current -= (dot_product / other_norm) * other
            projected.append(current)

        return np.asarray(np.sum(projected, axis=0).reshape(shape), dtype=float)


class MGDAMerger(GradientMerger):
    """Pareto-style gradient merger via Frank-Wolfe on the gradient Gram matrix."""

    def __init__(self, max_iter: int = 250, tolerance: float = 1e-8) -> None:
        """Initialize with Frank-Wolfe iteration limits."""
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        self.max_iter = max_iter
        self.tolerance = tolerance

    def merge(self, gradients: list[np.ndarray], weights: list[float] | np.ndarray | None = None) -> np.ndarray:
        """Merge gradients via Pareto-optimal convex combination."""
        shape, flattened = self._flatten_inputs(gradients, weights)
        if flattened.shape[0] == 1:
            return np.asarray(flattened[0].reshape(shape), dtype=float)

        gram = flattened @ flattened.T
        coefficients = self._solve_simplex_qp(gram)
        return np.asarray((coefficients @ flattened).reshape(shape), dtype=float)

    def _solve_simplex_qp(self, gram: np.ndarray) -> np.ndarray:
        """Solve ``min 0.5 * a^T G a`` over the probability simplex."""
        task_count = gram.shape[0]
        coefficients = np.full(task_count, 1.0 / task_count, dtype=float)

        for _ in range(self.max_iter):
            gradient = gram @ coefficients
            vertex_index = int(np.argmin(gradient))
            vertex = np.zeros(task_count, dtype=float)
            vertex[vertex_index] = 1.0

            direction = vertex - coefficients
            quadratic = float(direction @ gram @ direction)
            if quadratic <= self.tolerance:
                break

            step_size = float(-(direction @ gradient) / quadratic)
            step_size = min(max(step_size, 0.0), 1.0)
            updated = coefficients + (step_size * direction)
            if np.linalg.norm(updated - coefficients, ord=1) <= self.tolerance:
                coefficients = updated
                break
            coefficients = updated

        return coefficients


MERGERS: dict[GradientMergerName, type[GradientMerger]] = {
    "weighted_sum": WeightedSumMerger,
    "pcgrad": PCGradMerger,
    "mgda": MGDAMerger,
}


# =============================================================================
# Pre-merge and post-merge gradient transforms
# =============================================================================


def align_norms(grads: list[np.ndarray], mode: Literal["none", "unit", "match_first"]) -> list[np.ndarray]:
    """Align gradient norms before merging."""
    if mode == "none" or len(grads) <= 1:
        return grads
    if mode == "unit":
        return [g / (np.linalg.norm(g) + 1e-7) for g in grads]
    if mode == "match_first":
        target_norm = np.linalg.norm(grads[0])
        return [grads[0]] + [g * (target_norm / (np.linalg.norm(g) + 1e-7)) for g in grads[1:]]
    raise ValueError(f"Unknown norm_alignment mode: {mode}")


def normalize_gradient(
    gradient: np.ndarray, mode: Literal["unit", "sqrt_length"], zero_thr: float = 1e-7
) -> np.ndarray:
    """Normalize merged gradient before update.

    Args:
        gradient (np.ndarray): Merged gradient array (L, vocab_size).
        mode (Literal["unit", "sqrt_length"]): ``"unit"`` = L2 normalize to magnitude 1.0.
            ``"sqrt_length"`` = Germinal-compatible: ``g * sqrt(eff_L) / ||g||``
            where ``eff_L`` = positions with non-zero gradient norm.
        zero_thr (float): Threshold for considering a position's gradient as zero.
    """
    gn = np.linalg.norm(gradient)
    if gn <= zero_thr:
        return gradient

    if mode == "unit":
        return np.asarray(gradient / gn, dtype=float)

    if mode == "sqrt_length":
        per_position_norm = np.sum(gradient**2, axis=-1)
        eff_l = float(np.sum(per_position_norm > zero_thr))
        return np.asarray(gradient * np.sqrt(eff_l) / (gn + 1e-7), dtype=float)

    raise ValueError(f"Unknown normalize_mode: {mode}")


def adam_step(
    logits: np.ndarray,
    gradient: np.ndarray,
    lr: float,
    adam_m: list[np.ndarray],
    adam_v: list[np.ndarray],
    adam_t: list[int],
    idx: int,
    beta1: float,
    beta2: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply one Adam (or SGD) update step and return updated logits.

    Mutates ``adam_m[idx]``, ``adam_v[idx]``, and ``adam_t[idx]`` in place — the
    lists are the state container, not inputs to be preserved.
    """
    adam_t[idx] += 1

    if beta1 == 0.0 and beta2 == 0.0:
        return logits - lr * gradient

    adam_m[idx] = beta1 * adam_m[idx] + (1.0 - beta1) * gradient
    adam_v[idx] = beta2 * adam_v[idx] + (1.0 - beta2) * gradient**2

    m_hat = adam_m[idx] / (1.0 - beta1 ** adam_t[idx])
    v_hat = adam_v[idx] / (1.0 - beta2 ** adam_t[idx])

    return logits - lr * m_hat / (np.sqrt(v_hat) + eps)
