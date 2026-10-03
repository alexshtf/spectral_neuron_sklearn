"""Fit affine coefficients and amplitude for random spectral shapes."""

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import assert_all_finite

from ._loss import loss_and_gradient
from .model import _symmetric_matrices


def _sample_shapes(
    n_features: int, dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a uniform direction in the space of traceless symmetric pencils.

    Independent standard Gaussians in isometric packed coordinates have the
    same law as ``(G + G.T) / 2`` for an iid standard Gaussian matrix G.
    Project out each trace, then normalize the entire collection together.
    Requires ``dim > 1``.
    """
    i, j = np.tril_indices(dim)
    diagonal = i == j
    shapes = rng.standard_normal((n_features + 1, len(i)))
    shapes[:, diagonal] -= shapes[:, diagonal].mean(axis=1, keepdims=True)
    return shapes / np.linalg.norm(shapes)


def _fit_linear(design: np.ndarray, y: np.ndarray, loss: str) -> np.ndarray:
    """Fit an unregularized linear predictor; design includes its intercept."""
    match loss:
        case "squared_error":
            return lstsq(design, y)[0]
        case "absolute_error":
            # LAD dual: maximize y @ u, with design.T @ u = 0 and |u| <= 1.
            result = linprog(
                -y,
                A_eq=design.T,
                b_eq=np.zeros(design.shape[1]),
                bounds=(-1, 1),
                method="highs",
            )
            if not result.success:
                raise RuntimeError(f"Absolute-error initialization failed: {result.message}")
            return -result.eqlin.marginals
        case "log_loss":
            fit = LogisticRegression(
                C=np.inf, fit_intercept=False, solver="lbfgs", max_iter=1000, tol=1e-8
            ).fit(design, y)
            return fit.coef_[0]
        case _:
            raise ValueError(f"Unknown loss: {loss!r}")


def initialize(
    X: np.ndarray,
    y: np.ndarray,
    dim: int,
    eig_idx: int,
    loss: str,
    n_init: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float]]:
    """Return the best convex candidate and every candidate's training loss.

    For the odd-dimensional middle eigenvalue, signed amplitudes can be
    absorbed into the shapes. Otherwise, test both orientations of each draw
    and replace negative-amplitude fits with the best affine fit. ``n_init``
    counts candidate fits, including each orientation separately; an affine
    fallback is fitted once as needed. Size-one matrices use only that fit.
    """
    affine_design = np.column_stack((np.ones(len(X)), X))
    if dim == 1:
        linear = _fit_linear(affine_design, y, loss)
        value, _ = loss_and_gradient(affine_design @ linear, y, loss)
        return linear[:, None], [value]

    i, j = np.tril_indices(dim)
    identity = (i == j).astype(float)
    middle = 2 * eig_idx == dim - 1
    orientations = 1 if middle else 2
    design = np.column_stack((affine_design, np.zeros(len(X))))
    affine_fit = None
    candidate_losses = []
    best_loss = np.inf
    for start in range(0, n_init, orientations):
        shapes = _sample_shapes(X.shape[1], dim, rng)
        eigenvalues = np.linalg.eigvalsh(
            _symmetric_matrices(affine_design @ shapes, dim)
        )
        for orientation in range(min(orientations, n_init - start)):
            sign = 1 if orientation == 0 else -1
            index = eig_idx if sign == 1 else dim - 1 - eig_idx
            design[:, -1] = sign * eigenvalues[:, index]
            linear = _fit_linear(design, y, loss)
            if not middle and linear[-1] < 0:
                if affine_fit is None:
                    affine_fit = _fit_linear(affine_design, y, loss)
                linear = np.r_[affine_fit, 0.0]
            value, _ = loss_and_gradient(design @ linear, y, loss)
            assert_all_finite(value)
            candidate_losses.append(value)
            if value < best_loss:
                best_loss = value
                coefficients = linear[:-1, None] * identity + linear[-1] * sign * shapes
    return coefficients, candidate_losses
