from unittest.mock import Mock

import numpy as np
import pytest

import spectral_neuron.fitting as fitting
import spectral_neuron._initialization as initialization
from spectral_neuron import ConvexAdamFitter, SpectralModel
from spectral_neuron._fitting import loss_and_gradient, parameter_gradient
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


def test_adam_matches_independent_per_parameter_moment_updates(monkeypatch):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(9, 2))
    y = rng.normal(size=len(X))
    initial = rng.normal(size=(3, 6))
    initial_loss = loss_and_gradient(SpectralModel(initial, 3, 1)(X), y, "squared_error")[0]
    monkeypatch.setattr(
        fitting, "initialize", lambda *args: (initial.copy(), [initial_loss])
    )
    fitter = ConvexAdamFitter(
        dim=3, max_iter=3, batch_size=len(X), learning_rate=0.04,
        beta_1=0.6, beta_2=0.8, epsilon=0.03, tol=0, random_state=5,
    )

    model = fitter.fit(X, y)

    coefficients = initial.copy()
    first = np.zeros_like(coefficients)
    second = np.zeros_like(coefficients)
    for step in range(1, 4):
        _, gradient = parameter_gradient(X, y, coefficients, 3, 1, "squared_error")
        first = 0.6 * first + 0.4 * gradient
        second = 0.8 * second + 0.2 * gradient**2
        first_hat = first / (1 - 0.6**step)
        second_hat = second / (1 - 0.8**step)
        # MLPRegressor adds epsilon before bias correction of the second
        # moment, so its equivalent corrected denominator rescales epsilon.
        denominator = np.sqrt(second_hat) + 0.03 / np.sqrt(1 - 0.8**step)
        coefficients -= 0.04 * first_hat / denominator

    np.testing.assert_allclose(model.coefficients, coefficients, rtol=1e-12, atol=1e-12)


def test_loss_plateau_stops_moving_parameters_unless_tolerance_is_zero(monkeypatch):
    X, y = np.zeros((1, 1)), np.array([0.05])
    monkeypatch.setattr(
        fitting, "initialize", lambda *args: (np.zeros((2, 1)), [0.05**2])
    )
    options = dict(
        dim=1, max_iter=12, learning_rate=0.1, beta_1=0, beta_2=0,
        epsilon=1e-30, n_iter_no_change=2, random_state=4,
    )
    stopping = ConvexAdamFitter(**options, tol=1e-6)
    fixed_epochs = ConvexAdamFitter(**options, tol=0)

    stopping.fit(X, y)
    fixed_epochs.fit(X, y)

    # Parameters alternate between intercepts 0 and 0.1, while loss is
    # constant. This is loss-based patience, not parameter-motion stopping.
    np.testing.assert_allclose(stopping.loss_curve_, 0.05**2, atol=1e-17)
    assert stopping.n_iter_ == 2
    assert stopping.converged_
    assert fixed_epochs.n_iter_ == 12
    assert not fixed_epochs.converged_


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_convex_adam_end_to_end_is_reproducible_and_reports_actual_losses(loss):
    rng = np.random.default_rng(13)
    X = rng.uniform(-1, 1, size=(43, 2))
    y = (
        rng.binomial(1, 0.5 + 0.3 * X[:, 0] * X[:, 1])
        if loss == "log_loss"
        else X[:, 0] ** 2 - 0.5 * X[:, 1]
    )
    options = dict(dim=3, loss=loss, n_init=3, max_iter=4, batch_size=17, tol=0, random_state=8)
    first, second = ConvexAdamFitter(**options), ConvexAdamFitter(**options)

    model = first.fit(X, y)
    repeated = second.fit(X, y)

    np.testing.assert_array_equal(model.coefficients, repeated.coefficients)
    np.testing.assert_array_equal(first.loss_curve_, second.loss_curve_)
    assert first.initial_loss_ == pytest.approx(min(first.initialization_losses_))
    assert first.n_iter_ == 4
    assert len(first.loss_curve_) == first.n_iter_
    actual_loss = loss_and_gradient(model(X), y, loss)[0]
    assert first.loss_curve_[-1] == pytest.approx(actual_loss)
    assert np.isfinite(actual_loss)
