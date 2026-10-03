"""Convex random-shape initialization followed by ordinary minibatch Adam."""

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import gen_batches
from sklearn.utils.validation import assert_all_finite, check_scalar, check_X_y

from ._fitting import LOSSES, _objective, parameter_gradient
from ._initialization import initializations, initialize
from .model import SpectralModel, _resolve_eig_idx


class _AdamFitter:
    """Shared validation and per-candidate Adam updates."""

    def _prepare(self, X, y):
        eig_idx = _resolve_eig_idx(self.dim, self.eig_idx)
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}; got {self.loss!r}")
        check_scalar(self.n_init, "n_init", Integral, min_val=1)
        if self.batch_size != "auto":
            check_scalar(self.batch_size, "batch_size", Integral, min_val=1)
        for name, upper, boundaries in (
            ("learning_rate", np.inf, "neither"),
            ("epsilon", np.inf, "neither"),
            ("beta_1", 1, "left"),
            ("beta_2", 1, "left"),
        ):
            check_scalar(
                getattr(self, name), name, Real,
                min_val=0, max_val=upper, include_boundaries=boundaries,
            )
        assert_all_finite([self.learning_rate, self.epsilon, self.beta_1, self.beta_2])
        if self.random_state is not None:
            check_scalar(self.random_state, "random_state", Integral, min_val=0)
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        if self.loss == "log_loss" and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("log_loss requires both binary targets 0 and 1")
        batch_size = min(200 if self.batch_size == "auto" else self.batch_size, len(X))
        return X, y, eig_idx, batch_size

    def _adam_epochs(
        self, X, y, coefficients, eig_idx, batch_size, rng, n_iter
    ) -> Iterator[float]:
        """Update coefficients in place and yield each epoch's full training loss."""
        momentum = np.zeros_like(coefficients)
        mean_square = np.zeros_like(coefficients)
        step = 0
        for _ in range(n_iter):
            order = rng.permutation(len(X))
            for batch_slice in gen_batches(len(X), batch_size):
                batch = order[batch_slice]
                _, gradient = parameter_gradient(
                    X[batch], y[batch], coefficients, self.dim, eig_idx, self.loss
                )
                step += 1
                momentum *= self.beta_1
                momentum += (1.0 - self.beta_1) * gradient
                mean_square *= self.beta_2
                mean_square += (1.0 - self.beta_2) * gradient**2
                rate = (
                    self.learning_rate * np.sqrt(1.0 - self.beta_2**step)
                    / (1.0 - self.beta_1**step)
                )
                coefficients -= rate * momentum / (np.sqrt(mean_square) + self.epsilon)

            value = _objective(
                X, y, coefficients, self.dim, eig_idx, self.loss, batch_size
            )
            assert_all_finite(value)
            yield value


@dataclass
class ConvexAdamFitter(_AdamFitter):
    """Convex random-shape initialization followed by ordinary Adam.

    ``n_init`` counts convex candidate fits; opposite orientations count
    separately for noncentral eigenvalues. A required affine fallback is
    fitted once in addition to that budget. ``dim=1`` uses only an affine fit.
    Adam then refines the selected matrices with per-parameter moments.

    ``max_iter`` counts epochs; ``batch_size="auto"`` uses min(200, n_samples).
    Stop after ``n_iter_no_change`` epochs without an absolute improvement of
    at least ``tol`` over the best full training loss, including initialization.
    ``tol=0`` disables stopping. Adam's epsilon convention matches MLPRegressor.
    ``random_state`` seeds a local generator for initialization and shuffling.

    ``fit`` returns the final SpectralModel. Log loss requires both classes
    encoded as 0 and 1 and produces logits. Scale inputs and regression targets
    before fitting. All fitting and stopping decisions use training data.

    Diagnostics: ``initialization_losses_`` (one per candidate), ``initial_loss_``,
    ``loss_curve_`` (one per Adam epoch), ``n_iter_``, and ``converged_``. The last
    indicates loss-based stopping, not a certified optimum; exhausted budgets
    raise ConvergenceWarning unless ``tol=0``.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    n_init: int = 50
    learning_rate: float = 1e-3
    max_iter: int = 200
    batch_size: int | str = "auto"
    tol: float = 1e-4
    n_iter_no_change: int = 10
    random_state: int | None = None
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpectralModel:
        """Fit numeric arrays and return the final spectral model."""
        for name in ("max_iter", "n_iter_no_change"):
            check_scalar(getattr(self, name), name, Integral, min_val=1)
        check_scalar(self.tol, "tol", Real, min_val=0)
        assert_all_finite(self.tol)
        X, y, eig_idx, batch_size = self._prepare(X, y)
        rng = np.random.default_rng(self.random_state)
        coefficients, self.initialization_losses_ = initialize(
            X, y, self.dim, eig_idx, self.loss, self.n_init, rng
        )
        self.initial_loss_ = _objective(
            X, y, coefficients, self.dim, eig_idx, self.loss, batch_size
        )
        assert_all_finite(self.initial_loss_)
        best_loss = self.initial_loss_
        self.loss_curve_ = []
        self.converged_ = False
        no_improvement = 0
        for value in self._adam_epochs(
            X, y, coefficients, eig_idx, batch_size, rng, self.max_iter
        ):
            self.loss_curve_.append(value)
            if self.tol > 0.0:
                no_improvement = no_improvement + 1 if value > best_loss - self.tol else 0
                best_loss = min(best_loss, value)
                if no_improvement >= self.n_iter_no_change:
                    self.converged_ = True
                    break

        self.n_iter_ = len(self.loss_curve_)
        if self.tol > 0.0 and not self.converged_:
            warnings.warn(
                "Adam reached max_iter before the loss-based stopping criterion.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return SpectralModel(coefficients, self.dim, eig_idx)


@dataclass
class ConvexAdam2Fitter(_AdamFitter):
    """Refine every convex initialization, then select the best final model.

    Uses the same candidate construction and ``n_init`` budget as
    ConvexAdamFitter. Each candidate receives exactly ``n_iter`` full epochs
    of ordinary Adam with fresh moments. There is no tolerance, patience, or
    checkpoint selection. The smallest final full training loss wins.
    ``dim=1`` has just one affine initialization.

    Defaults: 50 candidates, 50 epochs each, learning rate 1e-2, and batches
    of min(200, n_samples). Initialization and shuffling use separate local
    random streams, so changing ``n_iter`` does not change the sampled shapes.
    Log loss requires both classes encoded as 0 and 1 and returns logits.
    Scale inputs and regression targets before fitting.

    Diagnostics: ``initialization_losses_`` and ``final_losses_`` in candidate
    order, ``best_init_`` (zero-based index), and the winner's ``initial_loss_``
    and ``loss_curve_``. ``n_iter_`` is the number of epochs per candidate.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    n_init: int = 50
    learning_rate: float = 1e-2
    n_iter: int = 50
    batch_size: int | str = "auto"
    random_state: int | None = None
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpectralModel:
        """Run every candidate for the fixed budget; return the best final model."""
        check_scalar(self.n_iter, "n_iter", Integral, min_val=1)
        X, y, eig_idx, batch_size = self._prepare(X, y)
        rng = np.random.default_rng(self.random_state)
        adam_rng = rng.spawn(1)[0]
        self.initialization_losses_ = []
        self.final_losses_ = []
        best_loss = np.inf
        for index, (coefficients, initial_loss) in enumerate(initializations(
            X, y, self.dim, eig_idx, self.loss, self.n_init, rng
        )):
            self.initialization_losses_.append(initial_loss)
            curve = list(self._adam_epochs(
                X, y, coefficients, eig_idx, batch_size, adam_rng, self.n_iter
            ))
            self.final_losses_.append(curve[-1])
            if curve[-1] < best_loss:
                best_loss = curve[-1]
                self.best_init_ = index
                self.initial_loss_ = initial_loss
                self.loss_curve_ = curve
                best_coefficients = coefficients
        self.n_iter_ = self.n_iter
        return SpectralModel(best_coefficients, self.dim, eig_idx)
