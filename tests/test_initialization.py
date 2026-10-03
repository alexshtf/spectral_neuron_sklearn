import numpy as np
import pytest

from spectral_neuron import SpectralModel
from spectral_neuron._initialization import _fit_affine, _sample_shapes, initialize
from spectral_neuron._loss import loss_and_gradient
from spectral_neuron._penalty import value


def test_shapes_have_balanced_feature_spectra_and_a_joint_unit_scale():
    state = np.random.get_state()
    shapes = _sample_shapes(2, 4, np.random.default_rng(19))
    matrices = SpectralModel(shapes, 4, 1).matrices
    eigenvalues = np.linalg.eigvalsh(matrices[1:])
    np.testing.assert_allclose(eigenvalues[:, 0], -eigenvalues[:, -1], atol=1e-15)
    assert value(shapes, 4) == pytest.approx(1)
    assert np.trace(matrices[0]) == pytest.approx(0, abs=1e-15)
    np.testing.assert_array_equal(shapes, _sample_shapes(2, 4, np.random.default_rng(19)))
    np.testing.assert_equal(state, np.random.get_state())


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_convex_affine_fit_handles_signed_coefficients_and_fixed_offsets(loss):
    if loss == "log_loss":
        x = np.repeat([-1.0, 0.0, 1.0], 4)
        y = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
        offset, expected = .8, [-.8, np.log(3)]
    else:
        x = np.arange(5.0) - 2
        offset = .3*x**2
        y = -2 - 3*x + offset
        y[-1] += 20
        expected = [2, 1] if loss == "squared_error" else [-2, -3]
    coefficients = _fit_affine(x[:, None], y, loss, offset=offset)
    np.testing.assert_allclose(coefficients, expected, atol=2e-5)


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error", "log_loss"])
def test_initialization_scores_the_reconstructed_regularized_model(loss):
    rng = np.random.default_rng(13)
    X = rng.uniform(-1, 1, size=(23, 2))
    y = rng.binomial(1, .5, len(X)) if loss == "log_loss" else X[:, 0]**2 - X[:, 1]
    coefficients, objectives, affine = initialize(
        X, y, 3, 0, loss, 3, np.random.default_rng(5), alpha=.1,
    )
    objective = loss_and_gradient(SpectralModel(coefficients, 3, 0)(X), y, loss)[0]
    assert objective + .1*value(coefficients, 3) == pytest.approx(min(objectives))
    diagonal = np.tril_indices(3)[0] == np.tril_indices(3)[1]
    assert np.count_nonzero(affine[:, ~diagonal]) == 0
    np.testing.assert_allclose(affine[:, diagonal], np.repeat(affine[:, :1], 3, axis=1))
