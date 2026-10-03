"""Convex initialization and full-batch proximal bundle fitting."""

from dataclasses import dataclass
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import assert_all_finite, check_scalar, check_X_y

from ._bundle import minimize
from ._initialization import initialize
from ._loss import LOSSES, loss_and_gradient
from ._penalty import prox as penalty_prox, value as penalty_value
from .model import SpectralModel, _resolve_eig_idx, _symmetric_matrices


@dataclass
class ProximalBundleFitter:
    """Fit exact convex losses with an operator-norm feature penalty.

    Standardize inputs and regression targets, convexly calibrate ``n_init``
    random nonlinear candidates, then refine their best penalized objective.
    The bundle retains 20 loss observations and two BFGS curvature pairs.
    Its spectral-norm prox can remove entire feature matrices exactly.

    ``alpha`` multiplies the sum of feature matrix operator norms in these
    standardized coordinates; the bias is unpenalized. ``max_iter`` counts
    trial steps, including rejected trials. Zero returns the better of the
    nonlinear initialization and the affine baseline. ``tol`` bounds relative
    bundle-model decrease; it is not a nonsmooth stationarity certificate.

    ``initialization_objectives_``, ``initial_objective_``, ``objective_curve_``
    and ``objective_`` use the standardized penalized objective. ``initial_loss_``,
    ``loss_curve_`` and ``loss_`` are data losses in original target units.
    Curves contain accepted nonlinear steps only. Initial values describe the
    nonlinear refinement start; ``used_affine_`` identifies a better final
    affine baseline, which may also win when ``max_iter=0``.

    ``n_iter_`` counts trials, ``n_accepted_`` accepted steps, and
    ``n_evaluations_`` optimizer loss/gradient calls, excluding initialization
    and diagnostics. ``model_decrease_`` and ``duality_gap_`` describe the last
    convex subproblem, or are None with no iterations. ``converged_`` means
    the model-decrease tolerance was met; ``message_`` records termination.
    Other termination with a positive budget raises ConvergenceWarning.

    All fitting uses training data only. ``fit`` returns an inference-only
    SpectralModel in original units. Direct logistic fitting requires both
    classes encoded as 0 and 1; the model returns logits. At eigenvalue ties
    use the eigenvector returned by eigh; at absolute-loss zeros use sign(0)=0.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    alpha: float = 0.0
    n_init: int = 50
    max_iter: int = 300
    tol: float = 1e-6
    random_state: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpectralModel:
        """Fit dense numeric arrays and return a model in original units."""
        eig_idx = _resolve_eig_idx(self.dim, self.eig_idx)
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}; got {self.loss!r}")
        check_scalar(self.n_init, "n_init", Integral, min_val=1)
        check_scalar(self.max_iter, "max_iter", Integral, min_val=0)
        for name in ("alpha", "tol"):
            check_scalar(getattr(self, name), name, Real, min_val=0)
            assert_all_finite(getattr(self, name))
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
        coefficients, objectives, affine = initialize(
            X, y, self.dim, eig_idx, self.loss, self.n_init,
            np.random.default_rng(self.random_state), self.alpha,
        )
        self.initialization_objectives_ = objectives
        design = np.column_stack((np.ones(len(X)), X))
        shape = coefficients.shape
        i, j = np.tril_indices(self.dim)
        factors = np.where(i == j, 1.0, np.sqrt(2.0))

        def objective(flat):
            matrices = _symmetric_matrices(design @ flat.reshape(shape), self.dim)
            values, vectors = np.linalg.eigh(matrices)
            value, derivative = loss_and_gradient(values[:, eig_idx], y, self.loss)
            v = vectors[:, :, eig_idx]
            sensitivity = v[:, i] * v[:, j] * factors
            gradient = design.T @ (derivative[:, None] * sensitivity)
            return value, gradient.ravel()

        def penalty(flat):
            return self.alpha * penalty_value(flat.reshape(shape), self.dim)

        def prox(flat, step):
            return penalty_prox(
                flat.reshape(shape), step * self.alpha, self.dim
            ).ravel()

        self.loss_curve_, self.objective_curve_ = [], []

        def record(_, loss, value):
            self.loss_curve_.append(loss * loss_scale)
            self.objective_curve_.append(value)

        initial_loss = objective(coefficients.ravel())[0]
        self.initial_loss_ = initial_loss * loss_scale
        self.initial_objective_ = initial_loss + penalty(coefficients.ravel())
        result = minimize(
            objective, prox, penalty, coefficients.ravel(),
            self.max_iter, self.tol, record,
        )
        affine_loss = loss_and_gradient(design @ affine[:, 0], y, self.loss)[0]
        affine_objective = affine_loss + penalty(affine.ravel())
        self.used_affine_ = bool(affine_objective < result.fun)
        coefficients = affine.copy() if self.used_affine_ else result.x.reshape(shape).copy()
        self.loss_ = (affine_loss if self.used_affine_ else result.loss) * loss_scale
        self.objective_ = affine_objective if self.used_affine_ else float(result.fun)
        self.n_iter_, self.n_evaluations_ = int(result.nit), int(result.nfev)
        self.n_accepted_ = len(self.objective_curve_)
        self.converged_, self.message_ = bool(result.success), str(result.message)
        self.model_decrease_, self.duality_gap_ = result.model_decrease_bound, result.model_gap

        coefficients[1:] /= scaler.scale_[:, None]
        coefficients[0] -= scaler.mean_ @ coefficients[1:]
        coefficients *= target_scale
        coefficients[0, i == j] += target_mean
        if self.max_iter and not self.converged_:
            warnings.warn(
                f"Proximal bundle stopped: {self.message_}", ConvergenceWarning, stacklevel=2
            )
        return SpectralModel(coefficients, self.dim, eig_idx)
