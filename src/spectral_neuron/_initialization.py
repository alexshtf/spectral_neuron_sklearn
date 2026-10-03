"""Fixed-scale spectral directions with convex affine calibration."""

import warnings

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.sparse import bmat, csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ._loss import loss_and_gradient
from ._penalty import SpectralPenalty
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


def _scalar_prox(coefficients, tau, penalty):
    """Prox of absolute values and anchored scalar group ranges."""
    result = coefficients.copy()
    values = coefficients[penalty.ungrouped]
    result[penalty.ungrouped] = np.sign(values) * np.maximum(np.abs(values) - tau, 0)
    for group in penalty.groups:
        values = coefficients[group]
        ordered = np.sort(np.maximum([values, -values], 0), axis=1)[:, ::-1]
        caps = np.maximum(((ordered.cumsum(axis=1) - tau)
                           / np.arange(1, len(group) + 1)).max(axis=1), 0)
        result[group] = np.clip(values, -caps[1], caps[0])
    return result


def _calibrate(design, y, loss, penalty, shapes=None, offset=0.0):
    """Fit identity shifts through linear epigraphs of each spectral norm."""
    if not penalty.groups or penalty.alpha == 0:
        return _fit_affine(design[:, 1:], y, loss, penalty.alpha, offset)

    n, q = design.shape
    E = np.concatenate((penalty.D, -penalty.D))
    block = np.tile(np.repeat(np.arange(len(penalty.blocks)),
                             [s.stop - s.start for s in penalty.blocks]), 2)
    R = np.eye(len(penalty.blocks))[block]
    h = np.zeros(len(E))
    if shapes is not None:
        edges = np.linalg.eigvalsh(_symmetric_matrices(penalty.D @ shapes, penalty.dim))
        h = np.r_[-edges[:, -1], edges[:, 0]]
    if loss == "absolute_error":
        # Dual LAD: |u| <= 1/n, E.T v = design.T u, R.T v = alpha.
        constraints = bmat([[-csr_matrix(design.T), csr_matrix(E.T)],
                            [None, csr_matrix(R.T)]], format="csr")
        result = linprog(
            np.r_[-(y - offset), h], A_eq=constraints,
            b_eq=np.r_[np.zeros(q), np.full(R.shape[1], penalty.alpha)],
            bounds=[(-1 / n, 1 / n)] * n + [(0, None)] * len(h), method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Absolute-error initialization failed: {result.message}")
        coefficients = result.eqlin.marginals[:q]
        gradient = -E.T @ result.x[n:]
    else:
        def objective(z):
            value, derivative = loss_and_gradient(offset + design @ z[:q], y, loss)
            return (value + penalty.alpha * z[q:].sum(),
                    np.r_[design.T @ derivative, np.full(R.shape[1], penalty.alpha)])

        initial = _fit_affine(design[:, 1:], y, loss, offset=offset)
        epigraph = np.zeros(R.shape[1])
        np.maximum.at(epigraph, block, E @ initial - h)
        result = minimize(
            objective, np.r_[initial, epigraph], jac=True, method="SLSQP",
            constraints=LinearConstraint(np.c_[E, -R], -np.inf, h),
            options={"ftol": 1e-11, "maxiter": 1000},
        )
        if not result.success:
            warnings.warn(f"Affine initialization stopped: {result.message}", ConvergenceWarning)
        coefficients = result.x[:q]
        gradient = design.T @ loss_and_gradient(offset + design @ coefficients, y, loss)[1]
    if shapes is None:
        # A proximal optimality evaluation recovers exact zeros from solver roundoff.
        step = n / (2 * np.sum(design**2))
        coefficients = _scalar_prox(coefficients - step * gradient, step * penalty.alpha, penalty)
    return coefficients


def initialize(
    X: np.ndarray,
    y: np.ndarray,
    dim: int,
    eig_idx: int,
    loss: str,
    n_init: int,
    rng: np.random.Generator,
    alpha: float = 0.0,
    penalty: SpectralPenalty | None = None,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """Return the best nonlinear start, candidate objectives, and affine fit.

    Normalize random pencils by their mixed norm/diameter penalty, then fit
    identity shifts with the nonlinear scale fixed. This calibration is
    convex because the extreme eigenvalues shift affinely. ``X`` contains
    only free columns; anchored group matrices have already been eliminated.
    """
    if penalty is None:
        penalty = SpectralPenalty(X.shape[1], dim, alpha, [])
    design = np.column_stack((np.ones(len(X)), X))
    i, j = np.tril_indices(dim)
    identity = (i == j).astype(float)
    linear = _calibrate(design, y, loss, penalty)
    affine = linear[:, None] * identity
    if dim == 1:
        value = loss_and_gradient(design @ linear, y, loss)[0]
        return affine, [value + penalty(affine.ravel())], affine

    objectives, best = [], np.inf
    for _ in range(n_init):
        shapes = _sample_shapes(X.shape[1], dim, rng)
        if penalty.groups:
            shapes /= penalty.unweighted_value(shapes.ravel())
        offset = np.linalg.eigvalsh(_symmetric_matrices(design @ shapes, dim))[:, eig_idx]
        linear = _calibrate(design, y, loss, penalty, shapes, offset)
        candidate = shapes + linear[:, None] * identity
        value = loss_and_gradient(offset + design @ linear, y, loss)[0]
        value += penalty(candidate.ravel())
        objectives.append(value)
        if value < best:
            best, coefficients = value, candidate
    return coefficients, objectives, affine
