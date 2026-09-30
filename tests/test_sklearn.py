import numpy as np
import pytest

from sklearn.base import clone, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_neuron import SpectralNeuron


@pytest.mark.parametrize("loss", ["squared_error", "log_loss"])
def test_sklearn_clone_pipeline_and_arbitrary_binary_labels(loss):
    X = np.linspace(-1, 1, 61)[:, None]
    y = (
        np.where(X[:, 0] > 0, "positive", "negative")
        if loss == "log_loss"
        else 0.2 + X[:, 0]
    )
    estimator = SpectralNeuron(
        dim=1, loss=loss, max_iter=400, learning_rate=0.04,
        tol=0, random_state=6,
    )
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    assert not hasattr(copied, "model_")
    pipeline = Pipeline([("scale", StandardScaler()), ("neuron", copied)]).fit(X, y)

    if loss == "log_loss":
        assert is_classifier(copied)
        np.testing.assert_array_equal(copied.classes_, ["negative", "positive"])
        assert np.mean(pipeline.predict(X) == y) > 0.97
        np.testing.assert_allclose(pipeline.predict_proba(X).sum(axis=1), 1)
    else:
        assert is_regressor(copied)
        np.testing.assert_allclose(pipeline.predict(X), y, atol=0.01)
