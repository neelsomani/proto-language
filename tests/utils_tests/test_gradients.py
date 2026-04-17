"""Tests for proto_language.utils.gradients module."""

import numpy as np

from proto_language.utils.gradients import MGDAMerger, normalize_gradient


class TestMGDAMerger:
    def test_equal_gradients_preserved(self) -> None:
        g = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert np.allclose(MGDAMerger().merge([g, g]), g)

    def test_opposing_gradients_cancel(self) -> None:
        g = np.array([[1.0, -1.0], [2.0, 0.5]])
        assert np.linalg.norm(MGDAMerger().merge([g, -g])) < 1e-6


class TestNormalizeGradient:
    def test_sqrt_length_formula_matches_germinal(self) -> None:
        """Verify ``g * sqrt(eff_L) / ||g||`` where ``eff_L`` counts positions with non-zero norm."""
        grad = np.zeros((5, 20))
        grad[0, 3] = 1.0
        grad[2, 7] = 2.0  # eff_L=2
        expected = grad * np.sqrt(2.0) / (np.linalg.norm(grad) + 1e-7)
        assert np.allclose(normalize_gradient(grad, mode="sqrt_length"), expected)

    def test_unit_vs_sqrt_length_differ(self) -> None:
        grad = np.random.default_rng(0).standard_normal((8, 20))
        unit = normalize_gradient(grad, mode="unit")
        sqrtl = normalize_gradient(grad, mode="sqrt_length")
        assert np.isclose(np.linalg.norm(unit), 1.0)
        assert np.linalg.norm(sqrtl) > 1.0
        assert not np.allclose(unit, sqrtl)
