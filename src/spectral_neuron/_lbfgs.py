"""Convex initialization and full-batch L-BFGS in whitened coordinates."""

from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import assert_all_finite, check_scalar, check_X_y

from ._fitting import LOSSES, loss_and_gradient
from ._initialization import initialize
from .model import SpectralModel, _resolve_eig_idx, _symmetric_matrices


def _smooth_loss(prediction, y, loss, epsilon):
    """Use exact smooth losses and the Moreau envelope of absolute error."""
    if loss != "absolute_error" or epsilon == 0:
        return loss_and_gradient(prediction, y, loss)
    residual = prediction - y
    absolute = np.abs(residual)
    clipped = np.minimum(absolute, epsilon)
    value = np.mean(absolute - clipped + clipped**2 / (2 * epsilon))
    return float(value), np.clip(residual / epsilon, -1, 1) / len(y)


def _weak_wolfe_lbfgs(fun, x, max_iter, tol, callback):
    """Identity-base L-BFGS with Lewis–Overton's doubling/bisection search."""
    pairs = deque(maxlen=20)
    value, gradient = fun(x)
    evaluations = 1
    message = "Iteration budget exhausted."
    for iteration in range(max_iter + 1):
        if max_iter and np.linalg.norm(gradient, np.inf) <= tol:
            message = "Numerical gradient tolerance reached."
            break
        if iteration == max_iter:
            break
        direction = gradient.copy()
        weights = []
        for s, y, rho in reversed(pairs):
            weights.append(rho * (s @ direction))
            direction -= weights[-1] * y
        for (s, y, rho), weight in zip(pairs, reversed(weights)):
            direction += s * (weight - rho * (y @ direction))
        direction = -direction
        slope = gradient @ direction
        if not slope < 0:
            message = "No descent direction."
            break
        lower, upper, step = 0.0, np.inf, 1.0
        for _ in range(60):
            trial = x + step * direction
            trial_value, trial_gradient = fun(trial)
            evaluations += 1
            if not trial_value < value + 1e-4 * step * slope:
                upper = step
            elif not trial_gradient @ direction > 0.9 * slope:
                lower = step
            else:
                break
            step = (lower + upper) / 2 if np.isfinite(upper) else 2 * lower
        else:
            message = "Weak-Wolfe line search failed."
            break
        s, y = trial - x, trial_gradient - gradient
        curvature = s @ y
        if not curvature > 0:
            message = "Nonpositive curvature."
            break
        pairs.append((s, y, 1 / curvature))
        x, value, gradient = trial, trial_value, trial_gradient
        callback(x)
    return OptimizeResult(
        x=x, fun=value, jac=gradient, nit=iteration, nfev=evaluations,
        success=message == "Numerical gradient tolerance reached.", message=message,
    )


@dataclass
class _LBFGSFitter:
    """Shared normalization, spectral derivatives, and model reconstruction."""

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    n_init: int = 50
    max_iter: int = 300
    tol: float = 1e-5
    random_state: int | None = None

    def _smoothing_width(self):
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpectralModel:
        """Fit dense numeric arrays and return an inference-only model."""
        eig_idx = _resolve_eig_idx(self.dim, self.eig_idx)
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}; got {self.loss!r}")
        check_scalar(self.n_init, "n_init", Integral, min_val=1)
        check_scalar(self.max_iter, "max_iter", Integral, min_val=0)
        check_scalar(self.tol, "tol", Real, min_val=0)
        assert_all_finite(self.tol)
        if self.random_state is not None:
            check_scalar(self.random_state, "random_state", Integral, min_val=0)
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        if self.loss == "log_loss" and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("log_loss requires both binary targets 0 and 1")

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        target_mean, target_scale = 0.0, 1.0
        if self.loss != "log_loss":
            target_mean, target_scale = float(y.mean()), float(y.std()) or 1.0
        y = (y - target_mean) / target_scale
        loss_scale = target_scale**2 if self.loss == "squared_error" else target_scale
        coefficients, losses = initialize(
            X, y, self.dim, eig_idx, self.loss, self.n_init,
            np.random.default_rng(self.random_state),
        )
        self.initialization_losses_ = [value * loss_scale for value in losses]

        design = np.column_stack((np.ones(len(X)), X))
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        keep = singular > singular[0] * max(design.shape) * np.finfo(float).eps
        # design @ transform = Z, with Z.T @ Z / n = I on the observed span.
        Z = u[:, keep] * np.sqrt(len(X))
        transform = vt[keep].T * (np.sqrt(len(X)) / singular[keep])
        initial = (singular[keep, None] * (vt[keep] @ coefficients)) / np.sqrt(len(X))
        i, j = np.tril_indices(self.dim)
        factors = np.where(i == j, 1.0, np.sqrt(2.0))
        self.loss_curve_ = []
        exact_loss = 0.0
        width = self._smoothing_width()

        def objective(flat):
            nonlocal exact_loss
            matrices = _symmetric_matrices(Z @ flat.reshape(initial.shape), self.dim)
            values, vectors = np.linalg.eigh(matrices)
            prediction = values[:, eig_idx]
            value, derivative = _smooth_loss(prediction, y, self.loss, width)
            exact_loss = loss_and_gradient(prediction, y, self.loss)[0] * loss_scale
            v = vectors[:, :, eig_idx]
            sensitivity = v[:, i] * v[:, j] * factors
            gradient = Z.T @ (derivative[:, None] * sensitivity)
            return value, gradient.ravel()

        objective(initial.ravel())
        self.initial_loss_ = exact_loss
        result = self._minimize(
            objective, initial.ravel(), lambda _: self.loss_curve_.append(exact_loss),
        )
        objective(result.x)
        self.loss_ = exact_loss
        self.optimization_loss_ = float(result.fun) * loss_scale
        self.smoothing_error_bound_ = width / 2 * target_scale
        self.n_iter_, self.n_evaluations_ = int(result.nit), int(result.nfev)
        self.converged_, self.message_ = bool(result.success), str(result.message)
        self.gradient_norm_ = float(np.max(np.abs(result.jac)))

        coefficients = transform @ result.x.reshape(initial.shape)
        coefficients[1:] /= scaler.scale_[:, None]
        coefficients[0] -= scaler.mean_ @ coefficients[1:]
        coefficients *= target_scale
        coefficients[0, i == j] += target_mean
        if self.max_iter and not self.converged_:
            warnings.warn(f"L-BFGS stopped: {self.message_}", ConvergenceWarning, stacklevel=2)
        return SpectralModel(coefficients, self.dim, eig_idx)


class ConvexLBFGSFitter(_LBFGSFitter):
    """Fit a spectral neuron without a learning rate or minibatch schedule.

    Standardize inputs and regression targets internally. Whiten the augmented
    input design, select the best convex initialization, then run full-batch
    L-BFGS with analytic spectral derivatives and its own line search. Return a
    SpectralModel in the original feature and target units. Numerical rank
    deficiency uses the minimum-norm extension in standardized coordinates.

    Squared and logistic losses are unchanged. Absolute error uses its Moreau
    envelope with width ``2 * tol`` in standardized target units. Its mean-loss
    discrepancy is at most ``tol * target_scale``; this bounds approximation
    error, not optimization error. No residual smoothing removes eigenvalue
    collisions. Derivatives require a simple selected eigenvalue.

    ``n_init`` has the same candidate budget as ConvexAdamFitter. ``max_iter``
    limits L-BFGS iterations, not epochs or line-search evaluations. ``tol`` is
    the gradient tolerance; the relative objective tolerance is ``tol**2``.
    Log loss requires both classes encoded as 0 and 1 and returns logits.

    ``initialization_losses_``, ``initial_loss_``, ``loss_curve_`` and ``loss_``
    report the exact requested training loss in original units. The possibly
    smoothed final objective is ``optimization_loss_``; its discrepancy bound
    is ``smoothing_error_bound_``. ``n_iter_``, ``n_evaluations_``, ``message_``,
    ``gradient_norm_`` and ``converged_`` report solver diagnostics. Convergence
    means a numerical stopping criterion, not a certificate of an optimum.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpectralModel:
        check_scalar(self.max_iter, "max_iter", Integral, min_val=1)
        check_scalar(self.tol, "tol", Real, min_val=0, include_boundaries="neither")
        return super().fit(X, y)

    def _smoothing_width(self):
        return 2 * self.tol if self.loss == "absolute_error" else 0.0

    def _minimize(self, objective, initial, callback):
        return minimize(
            objective, initial, jac=True, method="L-BFGS-B", callback=callback,
            options={"maxiter": self.max_iter, "gtol": self.tol, "ftol": self.tol**2},
        )


class ConvexLBFGS2Fitter(_LBFGSFitter):
    """Convex initialization and exact-loss, unscaled weak-Wolfe L-BFGS.

    Standardize inputs and regression targets, then whiten the augmented design.
    Refine the best of ``n_init=50`` convex candidates with full-batch analytic
    spectral derivatives, an identity base metric and 20 curvature pairs.
    The line search uses Armijo decrease and weak Wolfe, with doubling and
    bisection. All three losses are exact, including absolute error.

    ``max_iter=300`` counts accepted steps; zero returns the initialization.
    ``tol=1e-5`` bounds the gradient infinity norm in optimization coordinates;
    zero disables a positive threshold. This is a numerical stopping rule,
    not a nonsmooth stationarity certificate. Repeated selected eigenvalues
    use the eigenvector returned by eigh; absolute residuals use sign(0)=0.
    Neither choice guarantees a descent direction at a nondifferentiable point.

    ``fit`` returns a SpectralModel in original input and target units, using
    the last accepted iterate. Log loss requires binary targets 0 and 1 and
    returns logits. Initialization and optimization use training data only.

    ``initialization_losses_``, ``initial_loss_``, ``loss_curve_`` (one value
    per accepted step), and ``loss_`` report exact losses in original units.
    ``n_iter_``, ``n_evaluations_``, ``gradient_norm_``, ``converged_`` and
    ``message_`` distinguish numerical tolerance, budget exhaustion and failure.
    Unsuccessful optimization with a positive budget raises ConvergenceWarning.
    """

    def _minimize(self, objective, initial, callback):
        return _weak_wolfe_lbfgs(objective, initial, self.max_iter, self.tol, callback)
