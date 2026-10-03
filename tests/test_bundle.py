import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from spectral_neuron._bundle import _subproblem, minimize
from spectral_neuron._penalty import SpectralPenalty
from spectral_neuron.model import _symmetric_matrices


@pytest.mark.parametrize("tau, expected", [
    (.5, [-3.5, 1, 3]), (2, [-2.5, 1, 2.5]), (8, [0, 0, 0]),
])
def test_single_matrix_recovery_clips_spectra_and_zeroes_at_the_nuclear_norm(tau, expected):
    penalty = SpectralPenalty(1, 3, tau)
    coefficients = np.array([[2., 0, 2, 0, 0, 2], [-4., 0, 1, 0, 0, 3]])
    _, normal = penalty.project(penalty.apply(coefficients))
    fitted = penalty.recover(normal, coefficients).reshape(penalty.shape)
    np.testing.assert_allclose(_symmetric_matrices(fitted[1], 3), np.diag(expected), atol=1e-15)
    np.testing.assert_array_equal(fitted[0], coefficients[0])


def test_metric_bundle_gap_contains_an_independent_grouped_matrix_epigraph_optimum():
    rng = np.random.default_rng(11)
    centre = rng.normal(size=12)
    gradients, errors = rng.normal(scale=.3, size=(3, 12)), np.array([0., .1, .2])
    damping, alpha = 1.3, .17
    pairs, inverse = [], np.eye(12)
    for _ in range(2):
        s = rng.normal(size=12)
        y = np.arange(1, 13)*s
        rho = 1/(s@y)
        left = np.eye(12) - rho*np.outer(s, y)
        inverse = left@inverse@left.T + rho*np.outer(s, s)
        pairs.append((s, y, rho))
    metric = np.linalg.inv(inverse)
    penalty = SpectralPenalty(4, 2, alpha, [[3, 2, 1]])
    trial, predicted, weights, _, gap = _subproblem(
        centre, gradients, errors, damping, pairs, penalty, np.full(3, 1/3), None, 0,
    )

    # A separate primal epigraph: cut height, group diameter, ungrouped norm.
    def spectra(x):
        matrices = _symmetric_matrices(x.reshape(4, 3), 2)
        return np.linalg.eigvalsh([matrices[1], matrices[2],
                                  matrices[1]-matrices[2], matrices[3]])

    def objective(z):
        delta = z[:12] - centre
        return z[12] + alpha*(z[13]+z[14]) + damping*(delta@metric@delta)/2

    def constraints(z):
        eigenvalues = spectra(z[:12])
        bounds = np.array([z[13]]*3 + [z[14]])[:, None]
        return np.r_[z[12] - gradients@(z[:12]-centre) + errors,
                     (bounds-eigenvalues).ravel(), (bounds+eigenvalues).ravel()]

    def epigraph(x):
        norms = np.abs(spectra(x)).max(axis=1)
        return np.r_[x, max(gradients@(x-centre)-errors), norms[:3].max(), norms[3]]

    reference = scipy_minimize(
        objective, epigraph(centre), method="SLSQP",
        constraints={"type": "ineq", "fun": constraints},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert reference.success, reference.message
    primal = objective(epigraph(trial))
    assert primal-gap <= reference.fun + 1e-8 <= primal + 2e-8
    assert predicted == pytest.approx(penalty(centre)-primal, abs=1e-12)
    assert 0 <= gap <= .1*predicted
    assert weights.sum() == pytest.approx(1)


def test_rejected_trial_preserves_the_centre_and_counts_against_the_budget():
    accepted = []
    result = minimize(
        lambda x: (50*float((x[0]-2)**2), np.array([100*(x[0]-2), 0.])),
        SpectralPenalty(1, 1), np.zeros(2), 1, 0,
        lambda *args: accepted.append(args),
    )
    np.testing.assert_array_equal(result.x, [0, 0])
    assert result.fun == 200 and result.nit == 1 and result.nfev == 2
    assert accepted == [] and not result.success


def test_bundle_combines_opposite_absolute_loss_gradients():
    result = minimize(
        lambda x: (abs(x[0]), np.array([1. if x[0] >= 0 else -1., 0.])),
        SpectralPenalty(1, 1), np.array([.3, 0.]), 50, 1e-8,
    )
    assert result.fun < 1e-6
    assert result.success and result.model_decrease_bound <= 1e-8
