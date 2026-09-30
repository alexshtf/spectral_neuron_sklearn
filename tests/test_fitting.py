from itertools import product

import numpy as np
import pytest

from spectral_neuron import SpectralModel
from spectral_neuron._fitting import (
    AdamFitter,
    _initialize_coefficients,
    loss_and_gradient,
    parameter_gradient,
)


@pytest.mark.parametrize("feature_bound", [0.5, 5.0, 50.0])
@pytest.mark.parametrize("eig_idx", [0, 2, 4])
def test_initial_selected_eigenvalue_stays_gapped_over_feature_box(feature_bound, eig_idx):
    dim, n_features = 5, 3
    coefficients = _initialize_coefficients(
        n_features, dim, eig_idx, np.random.default_rng(9), feature_bound
    )
    matrices = SpectralModel(coefficients, dim, eig_idx).matrices
    corners = feature_bound * np.array(list(product((-1, 1), repeat=n_features)))
    pencils = matrices[0] + np.einsum("ni,ijk->njk", corners, matrices[1:])
    eigenvalues = np.linalg.eigvalsh(pencils)
    distances = np.abs(eigenvalues - eigenvalues[:, eig_idx, None])
    distances[:, eig_idx] = np.inf
    assert np.min(distances) >= 0.5 - 1e-12


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_spectral_parameter_gradient_matches_finite_differences(loss):
    rng = np.random.default_rng(12)
    X = rng.uniform(-0.5, 0.5, size=(7, 2))
    coefficients = rng.normal(size=(3, 6))
    prediction = SpectralModel(coefficients, dim=3, eig_idx=1)(X)
    y = (
        np.arange(len(X)) % 2
        if loss == "log_loss"
        else prediction + np.linspace(0.2, 0.8, len(X))
    )
    value, gradient = parameter_gradient(X, y, coefficients, 3, 1, loss)

    step = 1e-6
    numerical = np.empty_like(coefficients)
    for index in np.ndindex(coefficients.shape):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += step
        minus[index] -= step
        upper = loss_and_gradient(SpectralModel(plus, 3, 1)(X), y, loss)[0]
        lower = loss_and_gradient(SpectralModel(minus, 3, 1)(X), y, loss)[0]
        numerical[index] = (upper - lower) / (2 * step)

    np.testing.assert_allclose(value, loss_and_gradient(prediction, y, loss)[0])
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-7)


def test_losses_are_means_and_logistic_loss_is_stable_for_extreme_logits():
    prediction = np.array([1.0, 4.0])
    target = np.array([2.0, 2.0])
    value, gradient = loss_and_gradient(prediction, target, "squared_error")
    assert value == pytest.approx(2.5)
    np.testing.assert_allclose(gradient, [-1.0, 2.0])
    value, gradient = loss_and_gradient(prediction, target, "absolute_error")
    assert value == pytest.approx(1.5)
    np.testing.assert_allclose(gradient, [-0.5, 0.5])

    with np.errstate(over="raise", invalid="raise"):
        value, gradient = loss_and_gradient(
            np.array([-1000.0, 0.0, 1000.0]), np.array([1.0, 0.0, 0.0]), "log_loss"
        )
    assert value == pytest.approx((2000 + np.log(2)) / 3)
    np.testing.assert_allclose(gradient, [-1 / 3, 1 / 6, 1 / 3])


def test_first_adam_update_corrects_both_moments():
    fitter = AdamFitter(
        dim=1, max_iter=1, learning_rate=0.1, beta_1=0.8,
        beta_2=0.7, epsilon=0.2, random_state=4,
    )
    X = np.zeros((1, 1))
    model = fitter.fit(X, np.array([2.0]))
    # A0 starts at zero, so its squared-error gradient is -4. The first
    # bias-corrected moments are g and g², independent of the decay rates.
    np.testing.assert_allclose(model(X), [0.1 * 4 / (4 + 0.2)])


def test_zero_tolerance_disables_stopping_on_a_plateau():
    X, y = np.zeros((2, 1)), np.zeros(2)
    options = dict(dim=1, max_iter=12, n_iter_no_change=2, random_state=0)
    fixed_epochs = AdamFitter(**options, tol=0)
    early_stopping = AdamFitter(**options, tol=1e-6)
    fixed_epochs.fit(X, y)
    early_stopping.fit(X, y)
    assert fixed_epochs.n_iter_ == fixed_epochs.max_iter
    assert early_stopping.n_iter_ < early_stopping.max_iter
