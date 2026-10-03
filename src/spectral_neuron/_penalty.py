"""Operator-norm penalty and Frobenius prox for symmetric feature matrices."""

import numpy as np

from .model import _symmetric_matrices


def value(coefficients, dim):
    """Sum feature-matrix operator norms, excluding the first (bias) matrix."""
    eigenvalues = np.linalg.eigvalsh(_symmetric_matrices(coefficients[1:], dim))
    return float(np.max(np.abs(eigenvalues), axis=-1).sum())


def prox(coefficients, tau, dim):
    """Apply the prox of ``tau * value`` in Frobenius-isometric coordinates."""
    if tau == 0:
        return coefficients.copy()
    eigenvalues, vectors = np.linalg.eigh(_symmetric_matrices(coefficients[1:], dim))
    ordered = np.sort(np.abs(eigenvalues), axis=-1)[:, ::-1]
    thresholds = (np.cumsum(ordered, axis=-1) - tau) / np.arange(1, dim + 1)
    active = np.maximum(np.sum(ordered > thresholds, axis=-1), 1)
    cap = np.maximum(thresholds[np.arange(len(active)), active - 1], 0.0)
    clipped = np.clip(eigenvalues, -cap[:, None], cap[:, None])
    matrices = (vectors * clipped[:, None, :]) @ vectors.swapaxes(-1, -2)
    i, j = np.tril_indices(dim)
    result = coefficients.copy()
    result[1:] = matrices[:, i, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    return result
