import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from spectral_neuron._bundle import _subproblem, minimize
from spectral_neuron._penalty import prox, value
from spectral_neuron.model import _symmetric_matrices


def test_spectral_prox_satisfies_matrix_kkt_and_preserves_the_bias():
    rng = np.random.default_rng(8)
    coefficients = rng.normal(size=(3, 6))
    coefficients[2] *= .01
    fitted = prox(coefficients, .7, 3)
    matrices = _symmetric_matrices(fitted[1:], 3)
    subgradients = _symmetric_matrices((coefficients[1:] - fitted[1:])/.7, 3)
    nuclear = np.sum(np.abs(np.linalg.eigvalsh(subgradients)), axis=1)
    norms = np.max(np.abs(np.linalg.eigvalsh(matrices)), axis=1)
    assert np.all(nuclear <= 1 + 1e-12)
    np.testing.assert_allclose(np.sum(subgradients*matrices, axis=(1, 2)), norms, atol=1e-12)
    np.testing.assert_array_equal(fitted[0], coefficients[0])
    np.testing.assert_array_equal(fitted[2], 0)
    np.testing.assert_array_equal(prox(coefficients, 0, 3), coefficients)


def test_metric_bundle_subproblem_matches_an_independent_matrix_epigraph_solve():
    rng = np.random.default_rng(11)
    centre = np.array([.2, -.1, .3, 1.2, .4, -.5])
    gradients, errors = rng.normal(scale=.3, size=(3, 6)), np.array([0., .1, .2])
    damping, alpha = 1.3, .17
    pairs, inverse = [], np.eye(6)
    for _ in range(2):
        s = rng.normal(size=6)
        y = np.arange(1, 7)*s
        rho = 1/(s@y)
        left = np.eye(6) - rho*np.outer(s, y)
        inverse = left@inverse@left.T + rho*np.outer(s, s)
        pairs.append((s, y, rho))
    metric = np.linalg.inv(inverse)
    penalty = lambda x: alpha*value(x.reshape(2, 3), 2)
    proximal = lambda x, t: prox(x.reshape(2, 3), t*alpha, 2).ravel()
    trial, predicted, weights, gap = _subproblem(
        centre, gradients, errors, damping, pairs, proximal, penalty, np.full(3, 1/3),
    )

    # The independent primal has one cut-height and one matrix-norm epigraph variable.
    def objective(z):
        delta = z[:6] - centre
        return z[6] + alpha*z[7] + damping*(delta@metric@delta)/2

    def constraints(z):
        eigenvalues = np.linalg.eigvalsh(_symmetric_matrices(z[3:6], 2))
        return np.r_[z[6] - gradients@(z[:6] - centre) + errors,
                     z[7] - eigenvalues, z[7] + eigenvalues]

    start = np.r_[centre, 0., value(centre.reshape(2, 3), 2)]
    reference = scipy_minimize(
        objective, start, method="SLSQP", constraints={"type": "ineq", "fun": constraints},
        bounds=[(None, None)]*7 + [(0, None)], options={"ftol": 1e-12, "maxiter": 300},
    )
    assert reference.success, reference.message
    primal = objective(np.r_[trial, max(gradients@(trial-centre)-errors), penalty(trial)/alpha])
    assert primal == pytest.approx(reference.fun, abs=2e-7)
    assert predicted == pytest.approx(penalty(centre)-primal, abs=1e-12)
    assert primal-gap <= reference.fun + 1e-9
    assert 0 <= gap < 1e-6
    assert weights.sum() == pytest.approx(1)


def test_rejected_trial_preserves_the_centre_and_counts_against_the_budget():
    accepted = []
    result = minimize(
        lambda x: (50*float((x[0]-2)**2), 100*(x-2)),
        lambda x, _: x, lambda _: 0., np.zeros(1), 1, 0,
        lambda *args: accepted.append(args),
    )
    np.testing.assert_array_equal(result.x, [0])
    assert result.fun == 200 and result.nit == 1 and result.nfev == 2
    assert accepted == [] and not result.success


def test_bundle_combines_opposite_absolute_loss_gradients():
    result = minimize(
        lambda x: (abs(x[0]), np.where(x >= 0, 1., -1.)),
        lambda x, _: x, lambda _: 0., np.array([.3]), 50, 1e-8,
    )
    assert result.fun < 1e-6
    assert result.success and result.model_decrease_bound <= 1e-8
