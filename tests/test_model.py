import pickle

import numpy as np
import pytest

from spectral_neuron import SpectralModel


def test_model_evaluates_symmetric_path_and_preserves_leading_dimensions():
    # In packed coordinates, off-diagonal entries have Frobenius normalization.
    coefficients = np.array(
        [[2.0, np.sqrt(2), 4.0], [1.0, 0.0, -1.0], [0.0, np.sqrt(2), 1.0]]
    )
    matrices = np.array(
        [
            [[2.0, 1.0], [1.0, 4.0]],
            [[1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    model = SpectralModel(coefficients, dim=2, eig_idx=1)
    X = np.random.default_rng(4).normal(size=(2, 3, 2))
    path = matrices[0] + np.einsum("...i,ijk->...jk", X, matrices[1:])
    expected = np.linalg.eigvalsh(path)[..., 1]

    np.testing.assert_allclose(model.matrices, matrices)
    np.testing.assert_allclose(model(X), expected)
    assert model(X).shape == (2, 3)
    np.testing.assert_allclose(model(X[0, 0]), expected[0, 0])
    assert model.n_features == 2


def test_model_owns_parameters_and_round_trips_through_pickle():
    coefficients = np.array([[1.0], [2.0]])
    model = SpectralModel(coefficients, dim=1, eig_idx=0)
    coefficients[:] = 0
    assert not model.coefficients.flags.writeable
    with pytest.raises(ValueError):
        model.coefficients[0, 0] = 0
    np.testing.assert_allclose(model(np.array([[1.0], [2.0]])), [3.0, 5.0])
    restored = pickle.loads(pickle.dumps(model))
    assert not restored.coefficients.flags.writeable
    np.testing.assert_allclose(restored(np.array([[1.0], [2.0]])), [3.0, 5.0])


def test_model_requires_feature_last_inputs():
    model = SpectralModel(np.ones((2, 1)), dim=1, eig_idx=0)
    assert model(np.ones((2, 3, 1))).shape == (2, 3)
    with pytest.raises(ValueError):
        model(np.ones(3))
    with pytest.raises(ValueError):
        model(np.ones((4, 2)))
