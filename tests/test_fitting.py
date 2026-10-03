from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from spectral_neuron import LBFGSFitter, SpectralModel
import spectral_neuron.fitting as fitting
from spectral_neuron._loss import loss_and_gradient

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_lbfgs_improves_nonlinear_fit_and_reports_loss_in_original_units(loss):
    t = np.linspace(-1, 1, 61)
    X = np.column_stack((12 + 7*t, -4 + 3*np.cos(3*t)))
    y = (np.sin(7*t) > 0).astype(float) if loss == "log_loss" else 100 + 20*np.sin(3*t)
    fitter = LBFGSFitter(dim=5, loss=loss, n_init=4, max_iter=100, random_state=7)
    model = fitter.fit(X, y)
    exact, _ = loss_and_gradient(model(X), y, loss)
    assert exact < fitter.initial_loss_ * .9
    assert fitter.loss_ == pytest.approx(exact)
    assert fitter.loss_curve_[-1] == pytest.approx(exact)
    assert fitter.initial_loss_ == pytest.approx(min(fitter.initialization_losses_))
    assert np.all(np.diff([fitter.initial_loss_, *fitter.loss_curve_]) < 0)


def test_rank_deficient_design_with_more_features_than_rows_and_constant_column():
    t = np.array([-1.2, -.4, .3, 1.1])
    X = np.column_stack((20 + 3*t, 7*t, np.full(4, 5), t + 1, 2*t, 1 - t))
    fitter = LBFGSFitter(dim=1, n_init=2, random_state=7)
    model = fitter.fit(X, 200 + 12*t)
    np.testing.assert_allclose(model(X), 200 + 12*t, atol=1e-8)
    np.testing.assert_allclose(model.coefficients[3], 0, atol=1e-10)
    assert np.isfinite(model.coefficients).all()


def test_local_rng_is_reproducible_without_changing_global_numpy_state():
    X = np.linspace(-1, 1, 31)[:, None]
    options = dict(dim=3, n_init=3, max_iter=12, random_state=8)
    state = np.random.get_state()
    first = LBFGSFitter(**options).fit(X, np.sin(3*X[:, 0]))
    np.testing.assert_equal(state, np.random.get_state())
    second = LBFGSFitter(**options).fit(X, np.sin(3*X[:, 0]))
    np.testing.assert_array_equal(first.coefficients, second.coefficients)


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


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
@pytest.mark.parametrize("eig_idx", [0, 2])
def test_spectral_gradient_and_inverse_coordinate_maps(monkeypatch, eig_idx, loss):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(13, 2)) * [2, 7] + [20, -50]
    y = np.arange(len(X)) % 2 if loss == "log_loss" else np.linspace(120, 300, len(X))
    C = rng.normal(scale=.2, size=(3, 6))
    C[0, [0, 2, 5]] += [-2, 0, 3]
    def initialize(Xs, ys, *args):
        value = loss_and_gradient(SpectralModel(C, 3, eig_idx)(Xs), ys, loss)[0]
        return C.copy(), [value]
    def minimize(fun, x0, max_iter, tol, callback):
        value, gradient = fun(x0)
        direction = rng.normal(size=x0.size)
        direction /= np.linalg.norm(direction)
        h = 1e-6
        numerical = (fun(x0+h*direction)[0] - fun(x0-h*direction)[0]) / (2*h)
        np.testing.assert_allclose(gradient @ direction, numerical, rtol=1e-5, atol=1e-7)
        fun(x0)
        callback(x0)
        return SimpleNamespace(x=x0, fun=value, jac=gradient, nit=1, nfev=4,
                               success=True, message="controlled coordinate test")
    monkeypatch.setattr(fitting, "initialize", initialize)
    monkeypatch.setattr(fitting, "_weak_wolfe_lbfgs", minimize)
    model = LBFGSFitter(dim=3, eig_idx=eig_idx, loss=loss, n_init=1).fit(X, y)
    expected = SpectralModel(C, 3, eig_idx)(StandardScaler().fit_transform(X))
    if loss != "log_loss":
        expected = y.mean() + y.std()*expected
    np.testing.assert_allclose(model(X), expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("max_iter", [0, 3])
def test_constant_target_and_initialization_only_have_no_accepted_steps(max_iter):
    X = np.ones((5, 2)) * [4, -20]
    fitter = LBFGSFitter(max_iter=max_iter, n_init=2, random_state=7)
    model = fitter.fit(X, np.full(len(X), 12.0))
    np.testing.assert_allclose(model(X), 12)
    assert fitter.n_iter_ == 0 and fitter.loss_curve_ == []
    assert fitter.loss_ == fitter.initial_loss_ == 0
    assert fitter.converged_ == bool(max_iter)


def test_weak_wolfe_crosses_an_absolute_value_kink():
    result = fitting._weak_wolfe_lbfgs(
        lambda x: (abs(x[0]), np.sign(x)), np.array([.3]), 1, 0, lambda _: None,
    )
    assert result.x[0] < 0 and 0 < result.fun < .3
    assert result.nit == 1 and not result.success


def test_failed_line_search_preserves_the_last_accepted_state():
    accepted = []
    # At the minimum, +1 is a valid subgradient but yields no descent step.
    result = fitting._weak_wolfe_lbfgs(
        lambda x: (abs(x[0]), np.where(x >= 0, 1., -1.)),
        np.zeros(1), 3, 0, accepted.append,
    )
    np.testing.assert_array_equal(result.x, [0])
    np.testing.assert_array_equal(result.jac, [1])
    assert result.fun == 0 and result.nit == 0 and accepted == []
    assert not result.success and "line search failed" in result.message
