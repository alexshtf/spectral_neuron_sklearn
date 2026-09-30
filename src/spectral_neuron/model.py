"""Inference for a dense affine symmetric-matrix spectral neuron."""

from dataclasses import dataclass
from functools import cache
from numbers import Integral

import numpy as np
from sklearn.utils.validation import check_array, check_scalar


def _resolve_eig_idx(dim: int, eig_idx: int | None) -> int:
    check_scalar(dim, "dim", Integral, min_val=1)
    eig_idx = dim // 2 if eig_idx is None else eig_idx
    check_scalar(eig_idx, "eig_idx", Integral, min_val=0, max_val=dim - 1)
    return int(eig_idx)


@cache
def _symmetric_layout(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Index and scale the isometric lower-triangular parameterization."""
    i, j = np.tril_indices(dim)
    indices = np.empty((dim, dim), dtype=np.intp)
    indices[i, j] = np.arange(len(i))
    indices[j, i] = indices[i, j]
    scale = np.where(i == j, 1.0, 1.0 / np.sqrt(2.0))
    indices.flags.writeable = False
    scale.flags.writeable = False
    return indices, scale


def _symmetric_matrices(packed: np.ndarray, dim: int) -> np.ndarray:
    indices, scale = _symmetric_layout(dim)
    return (packed * scale)[..., indices]


@dataclass(frozen=True, eq=False)
class SpectralModel:
    """A fitted scalar neuron ``lambda_k(A0 + sum(x_i * A_i))``.

    ``coefficients`` contains the packed lower triangles of ``A0, A1, ...``.
    Off-diagonal coordinates are multiplied by ``sqrt(2)`` so their Euclidean
    norm equals the symmetric matrix's Frobenius norm. The model owns a
    read-only float64 copy of these parameters and has no training state.

    Eigenvalues are indexed in ascending order, starting at zero. Inputs use
    feature-last shape ``(..., n_features)``; outputs preserve leading axes.
    """

    coefficients: np.ndarray
    dim: int
    eig_idx: int

    def __post_init__(self) -> None:
        eig_idx = _resolve_eig_idx(self.dim, self.eig_idx)
        coefficients = check_array(
            self.coefficients, dtype=np.float64, copy=True, ensure_min_samples=2
        )
        width = self.dim * (self.dim + 1) // 2
        if coefficients.shape[1] != width:
            raise ValueError(
                f"coefficients must have shape (n_features + 1, {width}) "
                "with at least one feature"
            )
        coefficients.flags.writeable = False
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "eig_idx", eig_idx)

    def __reduce__(self):
        return type(self), (self.coefficients, self.dim, self.eig_idx)

    @property
    def n_features(self) -> int:
        return self.coefficients.shape[0] - 1

    @property
    def matrices(self) -> np.ndarray:
        """Return the full symmetric matrices ``[A0, A1, ..., An]``."""
        return _symmetric_matrices(self.coefficients, self.dim)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = check_array(
            X, dtype=np.float64, ensure_2d=False, allow_nd=True, ensure_min_samples=0
        )
        if X.shape[-1:] != (self.n_features,):
            raise ValueError(
                f"X must have shape (..., {self.n_features}); got {X.shape}"
            )
        packed = X @ self.coefficients[1:] + self.coefficients[0]
        return np.linalg.eigvalsh(_symmetric_matrices(packed, self.dim))[
            ..., self.eig_idx
        ]
