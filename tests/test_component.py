import pickle

import numpy as np
import pytest
from scipy.special import expit
from sklearn.exceptions import NotFittedError

from spectral_neuron import ProximalBundleFitter, SpectralModel, SpectralNeuron


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_linear_regression_and_binary_classification(loss):
    X = np.linspace(-1.0, 1.0, 81)[:, None]
    y = (X[:, 0] > 0.1).astype(int) if loss == "log_loss" else 0.6 + 1.25 * X[:, 0]
    estimator = SpectralNeuron(
        dim=1, loss=loss, n_init=2, max_iter=100, random_state=3,
    )
    transformed = estimator.fit_transform(X, y)
    assert transformed.shape == (len(X), 1)
    np.testing.assert_array_equal(transformed, estimator.transform(X))
    assert estimator.predict(X).shape == (len(X),)
    assert estimator.n_features_in_ == 1
    assert 0 <= estimator.n_iter_ <= estimator.max_iter
    assert np.isfinite(estimator.loss_curve_).all()

    if loss == "log_loss":
        logits = estimator.decision_function(X)
        probability = estimator.predict_proba(X)
        np.testing.assert_allclose(logits, estimator.transform(X)[:, 0])
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        np.testing.assert_allclose(probability[:, 1], expit(logits))
        np.testing.assert_array_equal(estimator.predict(X), (logits >= 0).astype(int))
        assert np.mean(estimator.predict(X) == y) > 0.97
    else:
        np.testing.assert_allclose(estimator.predict(X), estimator.transform(X)[:, 0])
        assert np.mean(np.abs(estimator.predict(X) - y)) < 0.02
        with pytest.raises(AttributeError):
            estimator.predict_proba(X)
        with pytest.raises(AttributeError):
            estimator.decision_function(X)


def test_dense_neuron_fits_nonlinear_function():
    X = np.linspace(-1, 1, 81)[:, None]
    y = np.sqrt(X[:, 0] ** 2 + 0.3**2)
    estimator = SpectralNeuron(
        dim=3, eig_idx=2, n_init=10, max_iter=300,
        tol=0, random_state=7,
    ).fit(X, y)
    assert np.mean((estimator.predict(X) - y) ** 2) < 5e-4
    assert estimator.loss_curve_[-1] < estimator.loss_curve_[0] / 20


def test_fitting_is_reproducible_and_pickle_preserves_predictions():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(37, 2))
    y = X[:, 0] - 0.3 * X[:, 1]
    options = dict(dim=2, n_init=3, max_iter=40, random_state=7)
    before = np.random.get_state()
    first = SpectralNeuron(**options).fit(X, y)
    after = np.random.get_state()
    second = SpectralNeuron(**options).fit(X, y)
    np.testing.assert_equal(before, after)
    np.testing.assert_array_equal(first.model_.coefficients, second.model_.coefficients)
    np.testing.assert_array_equal(first.loss_curve_, second.loss_curve_)
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(restored.predict(X), first.predict(X))


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_estimator_matches_direct_fitter_and_diagnostics(loss):
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, size=(23, 2))
    y = rng.binomial(1, 0.5, len(X)) if loss == "log_loss" else X[:, 0] ** 2 - X[:, 1]
    options = dict(dim=3, loss=loss, n_init=3, max_iter=4, tol=0, random_state=5)
    direct = ProximalBundleFitter(**options)
    model = direct.fit(X, y)
    estimator = SpectralNeuron(**options).fit(X, y)
    np.testing.assert_array_equal(estimator.model_.coefficients, model.coefficients)
    diagnostics = (
        "initialization_objectives_", "initial_loss_", "converged_",
        "message_", "n_evaluations_", "n_accepted_", "used_affine_",
        "objective_", "model_decrease_", "duality_gap_",
    )
    for name in (*diagnostics, "loss_curve_", "n_iter_"):
        np.testing.assert_equal(getattr(estimator, name), getattr(direct, name))


@pytest.mark.parametrize("loss", ["squared_error", "log_loss"])
def test_feature_strengths_are_raw_spectral_norms_of_feature_matrices(loss):
    matrices = np.array(
        [
            [[1000.0, 0.0], [0.0, 1000.0]],
            [[-5.0, 0.0], [0.0, 2.0]],
            [[0.0, 3.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    i, j = np.tril_indices(2)
    coefficients = matrices[:, i, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    estimator = SpectralNeuron(dim=2, loss=loss, max_iter=1, tol=0, random_state=7)
    estimator.fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    estimator.model_ = SpectralModel(coefficients, dim=2, eig_idx=1)
    strengths = estimator.feature_strengths_
    assert strengths.shape == (3,)
    np.testing.assert_allclose(strengths, [5.0, 3.0, 0.0])
    np.testing.assert_allclose(
        strengths, np.linalg.norm(matrices[1:], ord=2, axis=(-2, -1))
    )


def test_feature_strengths_require_fitting_and_refresh_after_refit():
    estimator = SpectralNeuron(dim=1, max_iter=1, tol=0, random_state=7)
    with pytest.raises(NotFittedError):
        estimator.feature_strengths_
    X = np.linspace(-1, 1, 6)[:, None]
    y = X[:, 0]
    estimator.fit(X, y)
    assert estimator.feature_strengths_.shape == (1,)
    np.testing.assert_allclose(
        estimator.feature_strengths_, np.abs(estimator.model_.coefficients[1:, 0])
    )
    estimator.fit(np.column_stack([X[:, 0], X[:, 0] ** 2]), y)
    assert estimator.feature_strengths_.shape == (2,)
    np.testing.assert_allclose(
        estimator.feature_strengths_, np.abs(estimator.model_.coefficients[1:, 0])
    )


def test_component_checks_fit_state_shape_and_logistic_targets():
    with pytest.raises((AttributeError, ValueError)):
        SpectralNeuron().transform(np.ones((3, 1)))
    with pytest.raises(ValueError):
        SpectralNeuron().fit(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        SpectralNeuron(loss="log_loss").fit(np.ones((3, 1)), np.array([0, 1, 2]))
    estimator = SpectralNeuron(dim=1, max_iter=1, tol=0).fit(np.ones((3, 1)), np.ones(3))
    with pytest.raises(ValueError):
        estimator.transform(np.ones((3, 2)))
