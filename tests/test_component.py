import pickle

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning

from spectral_neuron import SpectralNeuron


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_adam_fits_linear_regression_and_binary_classification(loss):
    X = np.linspace(-1.0, 1.0, 81)[:, None]
    y = (X[:, 0] > 0.1).astype(int) if loss == "log_loss" else 0.6 + 1.25 * X[:, 0]
    estimator = SpectralNeuron(
        dim=1, loss=loss, learning_rate=0.025, max_iter=600,
        batch_size=128, tol=0, n_iter_no_change=600, random_state=3,
    )
    assert estimator.fit(X, y) is estimator
    assert estimator.transform(X).shape == (len(X), 1)
    assert estimator.predict(X).shape == (len(X),)
    assert estimator.n_features_in_ == 1
    assert 0 < estimator.n_iter_ <= estimator.max_iter
    assert np.isfinite(estimator.loss_curve_).all()

    if loss == "log_loss":
        logits = estimator.decision_function(X)
        probability = estimator.predict_proba(X)
        np.testing.assert_allclose(logits, estimator.transform(X)[:, 0])
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        np.testing.assert_allclose(probability[:, 1], 1 / (1 + np.exp(-logits)))
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
        dim=3, eig_idx=2, learning_rate=0.025, max_iter=600,
        tol=0, n_iter_no_change=600, random_state=7,
    ).fit(X, y)
    assert np.mean((estimator.predict(X) - y) ** 2) < 5e-4
    assert estimator.loss_curve_[-1] < estimator.loss_curve_[0] / 20


def test_fitting_is_reproducible_and_pickle_preserves_predictions():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(37, 2))
    y = X[:, 0] - 0.3 * X[:, 1]
    options = dict(dim=2, max_iter=40, batch_size=9, random_state=7)
    np.random.seed(123)
    before = np.random.get_state()
    first = SpectralNeuron(**options).fit(X, y)
    after = np.random.get_state()
    second = SpectralNeuron(**options).fit(X, y)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(first.model_.coefficients, second.model_.coefficients)
    np.testing.assert_array_equal(first.loss_curve_, second.loss_curve_)
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(restored.predict(X), first.predict(X))


def test_fit_transform_and_parameter_api():
    X = np.linspace(-1, 1, 15)[:, None]
    estimator = SpectralNeuron(dim=1, max_iter=15, random_state=4)
    assert estimator.set_params(learning_rate=0.02, feature_bound="auto") is estimator
    assert estimator.get_params()["learning_rate"] == 0.02
    assert estimator.get_params()["feature_bound"] == "auto"
    transformed = estimator.fit_transform(X, X[:, 0])
    np.testing.assert_array_equal(transformed, estimator.transform(X))
    with pytest.raises(ValueError):
        estimator.set_params(nonexistent=1)


@pytest.mark.parametrize(
    "X, expected_bound",
    [
        (np.array([[-12.0, 2.0], [3.0, -6.0], [1.0, 4.0]]), 12.0),
        (np.zeros((4, 2)), 1.0),
    ],
)
def test_auto_feature_bound_matches_explicit_bound(X, expected_bound):
    y = np.linspace(-1, 1, len(X))
    options = dict(dim=3, max_iter=4, tol=0, random_state=7)
    automatic = SpectralNeuron(**options).fit(X, y)
    explicit = SpectralNeuron(**options, feature_bound=expected_bound).fit(X, y)
    assert automatic.feature_bound == "auto"
    assert automatic.feature_bound_ == explicit.feature_bound_ == expected_bound
    np.testing.assert_array_equal(automatic.model_.coefficients, explicit.model_.coefficients)
    np.testing.assert_array_equal(automatic.loss_curve_, explicit.loss_curve_)


def test_component_checks_fit_state_shape_and_logistic_targets():
    with pytest.raises((AttributeError, ValueError)):
        SpectralNeuron().transform(np.ones((3, 1)))
    with pytest.raises(ValueError):
        SpectralNeuron().fit(np.ones(3), np.ones(3))
    with pytest.warns(DataConversionWarning):
        SpectralNeuron(dim=1, max_iter=1).fit(np.ones((3, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError):
        SpectralNeuron(loss="log_loss").fit(np.ones((3, 1)), np.array([0, 1, 2]))
    estimator = SpectralNeuron(dim=1, max_iter=1).fit(np.ones((3, 1)), np.ones(3))
    with pytest.raises(ValueError):
        estimator.transform(np.ones((3, 2)))
