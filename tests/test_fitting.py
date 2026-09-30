from itertools import product

import numpy as np
import pytest

import spectral_neuron._fitting as fitting
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


@pytest.mark.parametrize("eig_idx", [0, 1, 2])
def test_grouped_adam_matches_dense_projector_updates(monkeypatch, eig_idx):
    # An independent reference works in the original matrix coordinates and
    # applies orthogonal projectors, rather than grouping packed coordinates.
    rng = np.random.default_rng(31)
    X = rng.uniform(-1, 1, size=(8, 2))
    y = rng.normal(size=len(X))
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    matrices = rng.normal(scale=0.1, size=(3, 3, 3))
    matrices += matrices.swapaxes(-1, -2)
    matrices[0] = (basis * [-2.0, 0.0, 3.0]) @ basis.T
    i, j = np.tril_indices(3)
    coefficients = matrices[:, i, j] * np.where(i == j, 1, np.sqrt(2))
    monkeypatch.setattr(fitting, "_initialize_coefficients", lambda *args: coefficients.copy())

    _, basis = np.linalg.eigh(matrices[0])
    selected = np.outer(basis[:, eig_idx], basis[:, eig_idx])
    remaining = np.eye(3) - selected

    def split(gradient):
        return np.stack(
            [
                selected @ gradient @ selected,
                selected @ gradient @ remaining + remaining @ gradient @ selected,
                remaining @ gradient @ remaining,
            ],
            axis=1,
        )

    options = dict(
        dim=3, eig_idx=eig_idx, max_iter=3, batch_size=len(X),
        learning_rate=0.04, beta_1=0.6, beta_2=0.8, epsilon=0.03,
        tol=0, random_state=4,
    )
    model = AdamFitter(**options).fit(X, y)
    momentum = np.zeros_like(matrices)
    variance = np.zeros((len(matrices), 3))
    augmented = np.column_stack([np.ones(len(X)), X])
    for step in range(1, 4):
        pencils = np.einsum("ni,ijk->njk", augmented, matrices)
        values, vectors = np.linalg.eigh(pencils)
        v = vectors[:, :, eig_idx]
        residual = values[:, eig_idx] - y
        gradient = np.einsum("ni,n,nj,nk->ijk", augmented, 2 * residual, v, v) / len(X)
        momentum = 0.6 * momentum + 0.4 * gradient
        # Dimensions of the three orthogonal symmetric-matrix subspaces.
        mean_squares = np.sum(split(gradient)**2, axis=(-2, -1)) / [1, 2, 3]
        variance = 0.8 * variance + 0.2 * mean_squares
        denominator = np.sqrt(variance / (1 - 0.8**step)) + 0.03
        direction = np.sum(split(momentum) / denominator[:, :, None, None], axis=1)
        matrices -= 0.04 * direction / (1 - 0.6**step)
    restored = basis @ model.matrices @ basis.T
    np.testing.assert_allclose(restored, matrices, rtol=1e-11, atol=1e-12)


def test_grouped_adam_is_invariant_to_internal_matrix_rotation(monkeypatch):
    rng = np.random.default_rng(19)
    X = rng.uniform(-1, 1, size=(9, 2))
    y = rng.normal(size=len(X))
    coefficients = _initialize_coefficients(2, 5, 2, rng, 1.0)
    initial = SpectralModel(coefficients, 5, 2)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    rotated = rotation @ initial.matrices @ rotation.T
    i, j = np.tril_indices(5)
    rotated_coefficients = rotated[:, i, j] * np.where(i == j, 1, np.sqrt(2))

    def fit_from(start):
        monkeypatch.setattr(fitting, "_initialize_coefficients", lambda *args: start.copy())
        _, basis = np.linalg.eigh(SpectralModel(start, 5, 2).matrices[0])
        fitter = AdamFitter(
            dim=5, eig_idx=2, max_iter=3, batch_size=3,
            epsilon=1e-5, tol=0, random_state=5,
        )
        model = fitter.fit(X, y)
        return basis @ model.matrices @ basis.T, fitter.loss_curve_, model(X)

    matrices, losses, prediction = fit_from(coefficients)
    rotated_matrices, rotated_losses, rotated_prediction = fit_from(rotated_coefficients)
    # A0 has repeated unselected eigenvalues, so its eigenbasis is not unique.
    # Sharing one scale across each subspace must remove that arbitrary choice.
    np.testing.assert_allclose(rotated_matrices, rotation @ matrices @ rotation.T, atol=1e-12)
    np.testing.assert_allclose(rotated_losses, losses, atol=1e-13)
    np.testing.assert_allclose(rotated_prediction, prediction, atol=1e-12)


def test_stationary_parameters_stop_after_patience_unless_tolerance_is_zero():
    X, y = np.zeros((2, 1)), np.zeros(2)
    options = dict(dim=1, max_iter=12, n_iter_no_change=2, random_state=0)
    fixed_epochs = AdamFitter(**options, tol=0)
    early_stopping = AdamFitter(**options, tol=1e-6)
    fixed_epochs.fit(X, y)
    early_stopping.fit(X, y)
    assert fixed_epochs.n_iter_ == fixed_epochs.max_iter
    assert early_stopping.n_iter_ == early_stopping.n_iter_no_change


def test_constant_loss_does_not_stop_oscillating_parameters():
    X, y = np.zeros((1, 1)), np.array([0.05])
    fitter = AdamFitter(
        dim=1, max_iter=12, learning_rate=0.1, beta_1=0, beta_2=0,
        epsilon=1e-30, tol=1e-6, n_iter_no_change=2, random_state=0,
    )
    model = fitter.fit(X, y)
    # With zero moments, Adam alternates A0 between 0 and 0.1. Both have
    # the same squared error for y=0.05, despite substantial parameter motion.
    np.testing.assert_allclose(fitter.loss_curve_, 0.05**2, rtol=0, atol=1e-17)
    np.testing.assert_allclose(model(X), 0, atol=1e-16)
    assert fitter.n_iter_ == fitter.max_iter


def test_small_parameter_steps_stop_even_while_loss_improves():
    X, y = np.zeros((1, 1)), np.array([1000.0])
    fitter = AdamFitter(
        dim=1, max_iter=12, learning_rate=1e-8,
        tol=1e-6, n_iter_no_change=2, random_state=0,
    )
    model = fitter.fit(X, y)
    assert 0 < model.coefficients[0, 0] <= fitter.n_iter_ * fitter.tol
    assert np.all(np.diff(fitter.loss_curve_) < -fitter.tol)
    assert fitter.n_iter_ == fitter.n_iter_no_change
