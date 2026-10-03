import numpy as np
import pytest
from sklearn.base import clone

from spectral_neuron import ProximalBundleFitter, SpectralNeuron
from spectral_neuron._penalty import SpectralPenalty
from spectral_neuron.model import _symmetric_matrices

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


@pytest.mark.parametrize("groups", [
    [[]], [[1]], [[1, 1]], [[1, 2], [2, 3]], [[0, 2]], [[1, 4]],
    [[1, 2.5]], [[True, 2]],
])
def test_group_validation_rejects_invalid_indices_or_partition(groups):
    with pytest.raises(ValueError, match="onehot_groups"):
        SpectralNeuron(onehot_groups=groups, max_iter=0).fit(np.ones((4, 3)), np.arange(4.))


def test_diameter_counts_nonreference_pairs_and_dual_projection_obeys_its_budget():
    penalty = SpectralPenalty(4, 2, .7, [[3, 1, 2]])
    coefficients = np.array([[8., 0, 8], [-2., 0, -2], [3., 0, 3], [.4, 0, .4]])
    np.testing.assert_array_equal(penalty.free, [0, 1, 2, 4])
    assert penalty(coefficients) == pytest.approx(.7*(5 + .4))
    rng = np.random.default_rng(8)
    dual = rng.normal(size=penalty.apply(coefficients).shape)
    assert np.sum(penalty.apply(coefficients)*dual) == pytest.approx(
        coefficients.ravel() @ penalty.adjoint(dual)
    )
    projected, normal = penalty.project(dual)
    np.testing.assert_allclose(projected + normal, dual, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(_symmetric_matrices(projected, 2))
    norms = np.abs(np.linalg.eigvalsh(_symmetric_matrices(normal, 2))).max(axis=1)
    for block in penalty.blocks:
        assert np.abs(eigenvalues[block]).sum() <= penalty.alpha + 1e-12
        assert np.sum(projected[block]*normal[block]) == pytest.approx(
            penalty.alpha*norms[block].max(), abs=1e-12,
        )
    np.testing.assert_allclose(penalty.recover(penalty.apply(coefficients), coefficients),
                               coefficients.ravel(), atol=1e-14)
    removed = penalty.recover(np.zeros_like(normal), coefficients).reshape(penalty.shape)
    np.testing.assert_array_equal(removed[1:], 0)
    np.testing.assert_array_equal(removed[0], coefficients[0])


def test_unpenalized_arbitrary_inputs_keep_first_listed_references_zero():
    X = np.random.default_rng(4).normal(size=(31, 5))*[2, 3, 7, 11, 5] + 20
    y = 100 + X @ [2, 0, -3, 1, 0]
    estimator = SpectralNeuron(dim=1, onehot_groups=[[2, 3, 1], [5, 4]], max_iter=0)
    copied = clone(estimator).fit(X, y)
    assert copied.get_params() == estimator.get_params()
    np.testing.assert_array_equal(copied.model_.coefficients[[2, 5]], 0)
    np.testing.assert_allclose(copied.predict(X), y, atol=1e-9)


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
@pytest.mark.parametrize("dim, alpha", [(1, .04), (3, 10.)])
def test_grouped_fit_matches_scalar_solution_or_removes_groups_exactly(loss, dim, alpha):
    levels = np.repeat(np.arange(3), 16)
    X = np.column_stack((np.eye(3)[levels], np.eye(2)[np.arange(48) % 2]))
    y = (np.arange(48) % 16 < (4 + 4*levels)).astype(float) if loss == "log_loss" else 100 + 20*(levels-1)
    fitter = ProximalBundleFitter(
        dim=dim, loss=loss, alpha=alpha, onehot_groups=[[2, 3, 1], [5, 4]],
        n_init=2, max_iter=50, random_state=7,
    )
    model = fitter.fit(X, y)
    np.testing.assert_array_equal(model.coefficients[[2, 4, 5]], 0)
    if dim == 3:
        np.testing.assert_array_equal(model.coefficients[1:], 0)
    else:
        if loss == "log_loss":
            p = .75 - 3*alpha
            expected = np.log(p/(1-p))*(levels-1)
        else:
            slope = 1 - 1.5*alpha/np.sqrt(1.5) if loss == "squared_error" else 1
            expected = 100 + 20*slope*(levels-1)
        np.testing.assert_allclose(model(X), expected, atol=2e-5)
    assert np.isfinite(fitter.objective_)
    assert fitter.objective_ <= fitter.initial_objective_ + 1e-12
