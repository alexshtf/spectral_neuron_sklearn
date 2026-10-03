from unittest.mock import Mock

import numpy as np
import pytest

import spectral_neuron._initialization as initialization
from spectral_neuron import SpectralModel
from spectral_neuron._initialization import _fit_linear, _sample_shapes, initialize


def test_sampled_shapes_are_jointly_normalized_traceless_and_reproducible():
    before = np.random.get_state()
    shapes = _sample_shapes(2, 4, np.random.default_rng(19))
    matrices = SpectralModel(shapes, 4, 1).matrices

    np.testing.assert_array_equal(shapes, _sample_shapes(2, 4, np.random.default_rng(19)))
    np.testing.assert_allclose(np.trace(matrices, axis1=-2, axis2=-1), 0, atol=1e-15)
    assert np.sum(matrices**2) == pytest.approx(1)
    np.testing.assert_equal(before, np.random.get_state())


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_convex_linear_fit_uses_the_requested_unregularized_objective(loss):
    if loss == "log_loss":
        x = np.repeat([-1.0, 0.0, 1.0], 4)
        y = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
        expected = np.array([0, np.log(3)])
    else:
        x = np.arange(5.0) - 2
        y = -2 - 3 * x
        y[-1] += 20
        expected = [2, 1] if loss == "squared_error" else [-2, -3]
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = _fit_linear(design, y, loss)
    # The L1 solution has negative intercept and slope: neither coefficient
    # may inherit linprog's default nonnegative variable bounds.
    np.testing.assert_allclose(coefficients, expected, atol=2e-5)


@pytest.mark.parametrize("eig_idx", [0, 1])
def test_orientations_reconstruct_predictions_with_an_odd_candidate_budget(monkeypatch, eig_idx):
    # Packed diagonal matrices diag(-2, 0, 2) and diag(2, -1, -1).
    shapes = np.array([[-2.0, 0, 0, 0, 0, 2], [2.0, 0, -1, 0, 0, -1]])
    shapes /= np.linalg.norm(shapes)
    sample = Mock(return_value=shapes)
    monkeypatch.setattr(initialization, "_sample_shapes", sample)
    X = np.linspace(-2, 2, 41)[:, None]
    opposite_feature = SpectralModel(-shapes, 3, eig_idx)(X)
    y = 0.4 - 0.8 * X[:, 0] + 2 * opposite_feature

    coefficients, losses = initialize(
        X, y, 3, eig_idx, "squared_error", 3, np.random.default_rng(5)
    )

    # For a noncentral eigenvalue, negating the original selected feature
    # would be wrong; the opposite orientation uses the opposite eigenvalue.
    np.testing.assert_allclose(SpectralModel(coefficients, 3, eig_idx)(X), y, atol=1e-12)
    assert len(losses) == 3
    assert sample.call_count == (2 if eig_idx == 0 else 3)
    if eig_idx == 0:
        assert losses[1] < losses[0]
    else:
        assert losses[0] < 1e-24


def test_negative_nonmiddle_amplitude_refits_the_affine_baseline(monkeypatch):
    shapes = _sample_shapes(2, 3, np.random.default_rng(5))
    monkeypatch.setattr(initialization, "_sample_shapes", lambda *args: shapes.copy())
    X = np.random.default_rng(9).normal(size=(23, 2))
    y = 0.2 + X[:, 0] - 2 * SpectralModel(shapes, 3, 0)(X)
    design = np.column_stack([np.ones(len(X)), X])
    affine_prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]

    coefficients, losses = initialize(
        X, y, 3, 0, "squared_error", 1, np.random.default_rng(5)
    )

    np.testing.assert_allclose(SpectralModel(coefficients, 3, 0)(X), affine_prediction)
    assert losses[0] == pytest.approx(np.mean((affine_prediction - y) ** 2))


def test_size_one_initialization_is_an_affine_fit():
    X = np.linspace(-1, 1, 17)[:, None]
    y = 2.0 - 0.7 * X[:, 0]
    coefficients, losses = initialize(
        X, y, 1, 0, "squared_error", 10, np.random.default_rng(7)
    )

    np.testing.assert_allclose(coefficients, [[2.0], [-0.7]], atol=1e-12)
    assert max(losses) < 1e-24

