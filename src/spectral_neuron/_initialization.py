"""Fixed-scale spectral directions with convex affine calibration."""

import warnings

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import linprog, minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ._loss import loss_and_gradient
from .model import _symmetric_matrices


def _sample_shapes(
    n_features: int, dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw centered feature spectra with operator norms summing to one.

    Gaussian isometric coordinates give symmetric matrices. Remove their
    traces, center the feature spectra, then scale the whole pencil together.
    The bias stays traceless. Requires ``dim > 1``.
    """
    i, j = np.tril_indices(dim)
    diagonal = i == j
    shapes = rng.standard_normal((n_features + 1, len(i)))
    shapes[:, diagonal] -= shapes[:, diagonal].mean(axis=1, keepdims=True)
    edges = np.linalg.eigvalsh(_symmetric_matrices(shapes[1:], dim))[:, [0, -1]]
    shapes[1:, diagonal] -= edges.mean(axis=1, keepdims=True)
    return shapes / (np.diff(edges, axis=1).sum() / 2)


def _fit_affine(
    X: np.ndarray,
    y: np.ndarray,
    loss: str,
    alpha: float = 0.0,
    offset: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Fit mean loss(offset + intercept + X beta) + alpha ||beta||_1."""
    design = np.column_stack((np.ones(len(X)), X))
    if alpha == 0 and loss == "squared_error":
        return lstsq(design, y - offset)[0]
    if alpha == 0 and loss == "log_loss" and np.ndim(offset) == 0 and offset == 0:
        return LogisticRegression(
            C=np.inf, fit_intercept=False, solver="lbfgs", max_iter=1000, tol=1e-8
        ).fit(design, y).coef_[0]

    split = np.column_stack((X, -X))
    if loss == "absolute_error":
        # LAD dual: |u| <= 1/n, sum(u) = 0, and |X.T @ u| <= alpha.
        result = linprog(
            -(y - offset), A_ub=split.T, b_ub=np.full(split.shape[1], alpha),
            A_eq=np.ones((1, len(y))), b_eq=[0.0],
            bounds=(-1 / len(y), 1 / len(y)), method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Absolute-error initialization failed: {result.message}")
        intercept, parts = -result.eqlin.marginals[0], -result.ineqlin.marginals
    else:
        def objective(coefficients):
            value, derivative = loss_and_gradient(
                offset + coefficients[0] + split @ coefficients[1:], y, loss
            )
            return (value + alpha * coefficients[1:].sum(),
                    np.r_[derivative.sum(), split.T @ derivative + alpha])

        result = minimize(
            objective, np.zeros(split.shape[1] + 1), jac=True, method="L-BFGS-B",
            bounds=[(None, None)] + [(0, None)] * split.shape[1],
            options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            warnings.warn(f"Affine initialization stopped: {result.message}", ConvergenceWarning)
        intercept, parts = result.x[0], result.x[1:]
    return np.r_[intercept, parts[:X.shape[1]] - parts[X.shape[1]:]]


def initialize(
    X: np.ndarray,
    y: np.ndarray,
    dim: int,
    eig_idx: int,
    loss: str,
    n_init: int,
    rng: np.random.Generator,
    alpha: float = 0.0,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """Return the best nonlinear start, candidate objectives, and affine fit.

    Each random pencil has centered feature spectra and total feature norm
    one. Keep that nonlinear scale fixed and fit only identity shifts. Then
    ``||p_i I + S_i||_op = |p_i| + ||S_i||_op``, so affine calibration is
    convex and every recorded objective includes the exact norm penalty.
    Size-one matrices need only the regularized affine fit.
    """
    design = np.column_stack((np.ones(len(X)), X))
    i, j = np.tril_indices(dim)
    identity = (i == j).astype(float)
    linear = _fit_affine(X, y, loss, alpha)
    affine = linear[:, None] * identity
    if dim == 1:
        value = loss_and_gradient(design @ linear, y, loss)[0]
        return affine, [value + alpha * np.abs(linear[1:]).sum()], affine

    objectives, best = [], np.inf
    for _ in range(n_init):
        shapes = _sample_shapes(X.shape[1], dim, rng)
        offset = np.linalg.eigvalsh(_symmetric_matrices(design @ shapes, dim))[:, eig_idx]
        linear = _fit_affine(X, y, loss, alpha, offset)
        value = loss_and_gradient(offset + design @ linear, y, loss)[0]
        value += alpha * (np.abs(linear[1:]).sum() + 1.0)
        objectives.append(value)
        if value < best:
            best, coefficients = value, shapes + linear[:, None] * identity
    return coefficients, objectives, affine
