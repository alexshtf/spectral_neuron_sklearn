from unittest.mock import Mock

import numpy as np
import pytest

import spectral_neuron.fitting as fitting
from spectral_neuron import ConvexAdam2Fitter, ConvexAdamFitter
from spectral_neuron._fitting import loss_and_gradient


def test_all_starts_get_independent_adam_and_selection_uses_final_loss(monkeypatch):
    X, y = np.zeros((1, 1)), np.zeros(1)
    starts = [np.array([[value], [0.0]]) for value in (0.1, 0.2)]
    monkeypatch.setattr(
        fitting, "initializations",
        lambda *args: ((start.copy(), start[0, 0] ** 2) for start in starts),
    )
    fitter = ConvexAdam2Fitter(dim=1, n_init=2, n_iter=3, learning_rate=1, random_state=4)
    model = fitter.fit(X, y)
    references, models = [], []
    for start in starts:
        monkeypatch.setattr(
            fitting, "initialize", lambda *args: (start.copy(), [start[0, 0] ** 2])
        )
        reference = ConvexAdamFitter(
            dim=1, max_iter=3, learning_rate=1, tol=0, random_state=4
        )
        models.append(reference.fit(X, y))
        references.append(reference.loss_curve_)

    # Start 0 is best initially and at its best checkpoint, but loses at
    # the final iterate. Start 1 must win even though its last step worsens.
    assert min(references[0]) < min(references[1])
    assert references[1][-1] > references[1][-2]
    assert fitter.best_init_ == 1
    assert fitter.initial_loss_ == pytest.approx(0.2**2)
    np.testing.assert_allclose(fitter.final_losses_, np.array(references)[:, -1])
    np.testing.assert_allclose(fitter.loss_curve_, references[1])
    np.testing.assert_allclose(model.coefficients, models[1].coefficients)


def test_plateau_does_not_shorten_any_candidates_epoch_budget(monkeypatch):
    monkeypatch.setattr(
        fitting, "initializations",
        lambda *args: ((np.zeros((2, 1)), 0.0) for _ in range(2)),
    )
    gradient = Mock(wraps=fitting.parameter_gradient)
    monkeypatch.setattr(fitting, "parameter_gradient", gradient)
    fitter = ConvexAdam2Fitter(dim=1, n_init=2, n_iter=12, random_state=3)
    fitter.fit(np.zeros((1, 1)), np.zeros(1))
    assert gradient.call_count == 24
    assert fitter.n_iter_ == len(fitter.loss_curve_) == 12
    np.testing.assert_array_equal(fitter.final_losses_, [0, 0])


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_real_candidates_are_reproducible_and_independent_of_training_budget(loss):
    rng = np.random.default_rng(13)
    X = rng.uniform(-1, 1, size=(23, 2))
    y = rng.binomial(1, 0.5, len(X)) if loss == "log_loss" else X[:, 0] ** 2 - X[:, 1]
    options = dict(dim=3, loss=loss, n_init=3, n_iter=2, batch_size=9, random_state=7)
    first, repeated = ConvexAdam2Fitter(**options), ConvexAdam2Fitter(**options)
    model = first.fit(X, y)
    np.testing.assert_array_equal(model.coefficients, repeated.fit(X, y).coefficients)
    np.testing.assert_array_equal(first.final_losses_, repeated.final_losses_)
    longer = ConvexAdam2Fitter(**(options | {"n_iter": 3}))
    longer.fit(X, y)
    np.testing.assert_array_equal(first.initialization_losses_, longer.initialization_losses_)
    assert len(first.final_losses_) == len(first.initialization_losses_) == 3
    assert first.best_init_ == np.argmin(first.final_losses_)
    assert first.initial_loss_ == first.initialization_losses_[first.best_init_]
    actual = loss_and_gradient(model(X), y, loss)[0]
    assert actual == pytest.approx(first.final_losses_[first.best_init_])
    assert actual == pytest.approx(first.loss_curve_[-1])
    assert first.n_iter_ == len(first.loss_curve_) == 2
