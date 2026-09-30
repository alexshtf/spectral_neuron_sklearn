"""Analytical spectral gradients and a NumPy Adam fitter."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.special import expit
from sklearn.utils.validation import assert_all_finite, check_scalar

from .model import SpectralModel, _resolve_eig_idx, _symmetric_matrices


LOSSES = ("squared_error", "absolute_error", "log_loss")


def loss_and_gradient(
    prediction: np.ndarray, y: np.ndarray, loss: str
) -> tuple[float, np.ndarray]:
    """Return mean loss and its derivative with respect to the predictions."""
    residual = prediction - y
    match loss:
        case "squared_error":
            return float(np.mean(residual**2)), 2.0 * residual / prediction.size
        case "absolute_error":
            # Choose the zero subgradient where the residual vanishes.
            return float(np.mean(np.abs(residual))), np.sign(residual) / prediction.size
        case "log_loss":
            values = np.logaddexp(0.0, (1.0 - 2.0 * y) * prediction)
            return float(np.mean(values)), (expit(prediction) - y) / prediction.size
        case _:
            raise ValueError(f"loss must be one of {LOSSES}; got {loss!r}")


def parameter_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    dim: int,
    eig_idx: int,
    loss: str,
) -> tuple[float, np.ndarray]:
    """Differentiate the mean objective in isometric packed coordinates.

    For a simple eigenvalue, its matrix derivative is ``v v.T``, where ``v``
    is the selected normalized eigenvector. At a repeated selected eigenvalue,
    the derivative is not unique; this uses the eigenvector returned by eigh.
    """
    packed = X @ coefficients[1:] + coefficients[0]
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric_matrices(packed, dim))
    value, derivative = loss_and_gradient(eigenvalues[..., eig_idx], y, loss)
    v = eigenvectors[..., :, eig_idx]
    i, j = np.tril_indices(dim)
    sensitivity = v[:, i] * v[:, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    sensitivity *= derivative[:, None]
    gradient = np.empty_like(coefficients)
    gradient[0] = sensitivity.sum(axis=0)
    gradient[1:] = X.T @ sensitivity
    return value, gradient


def _initialize_coefficients(
    n_features: int,
    dim: int,
    eig_idx: int,
    rng: np.random.Generator,
    feature_bound: float,
) -> np.ndarray:
    """Preserve the paper's gapped base and jittered identity feature matrices."""
    i, j = np.tril_indices(dim)
    diagonal = i == j
    coefficients = np.zeros((n_features + 1, len(i)), dtype=np.float64)
    bound = n_features**-0.5
    identity = rng.uniform(-bound, bound, n_features)
    jitter_bound = 1.0 / (4 * n_features * feature_bound)
    coefficients[1:, diagonal] = identity[:, None] + rng.uniform(
        -jitter_bound, jitter_bound, (n_features, dim)
    )

    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    spectrum = np.sign(np.arange(dim) - eig_idx)
    base = (q * spectrum) @ q.T
    coefficients[0] = base[i, j] * np.where(diagonal, 1.0, np.sqrt(2.0))
    return coefficients


def _objective(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    dim: int,
    eig_idx: int,
    loss: str,
    batch_size: int,
) -> float:
    total = 0.0
    for start in range(0, len(X), batch_size):
        stop = start + batch_size
        packed = X[start:stop] @ coefficients[1:] + coefficients[0]
        prediction = np.linalg.eigvalsh(_symmetric_matrices(packed, dim))[
            ..., eig_idx
        ]
        value, _ = loss_and_gradient(prediction, y[start:stop], loss)
        total += len(prediction) * value
    return total / len(X)


@dataclass
class AdamFitter:
    """Fit one spectral neuron with shuffled minibatches and internal Adam.

    ``max_iter`` counts epochs. ``tol`` and ``n_iter_no_change`` stop fitting
    after the full training objective stops improving; ``tol=0`` disables this
    stopping criterion. The returned model is the final iterate; the fitter
    does not select or restore checkpoints.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    feature_bound: float = 5.0
    learning_rate: float = 0.01
    max_iter: int = 200
    batch_size: int = 128
    tol: float = 1e-6
    n_iter_no_change: int = 10
    random_state: int | None = None
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    def _validate_parameters(self) -> int:
        eig_idx = _resolve_eig_idx(self.dim, self.eig_idx)
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}; got {self.loss!r}")
        for name in ("max_iter", "batch_size", "n_iter_no_change"):
            check_scalar(getattr(self, name), name, Integral, min_val=1)
        for name, upper, boundaries in (
            ("feature_bound", np.inf, "neither"),
            ("learning_rate", np.inf, "neither"),
            ("epsilon", np.inf, "neither"),
            ("tol", np.inf, "left"),
            ("beta_1", 1, "left"),
            ("beta_2", 1, "left"),
        ):
            check_scalar(
                getattr(self, name), name, Real,
                min_val=0, max_val=upper, include_boundaries=boundaries,
            )
        assert_all_finite(
            [
                self.feature_bound, self.learning_rate, self.epsilon,
                self.tol, self.beta_1, self.beta_2,
            ]
        )
        if self.random_state is not None:
            check_scalar(self.random_state, "random_state", Integral, min_val=0)
        return eig_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpectralModel:
        """Fit arrays already validated and encoded by the public estimator."""
        eig_idx = self._validate_parameters()
        rng = np.random.default_rng(self.random_state)
        coefficients = _initialize_coefficients(
            X.shape[1], self.dim, eig_idx, rng, self.feature_bound
        )
        momentum = np.zeros_like(coefficients)
        variance = np.zeros_like(coefficients)
        self.loss_curve_ = []
        best_loss = np.inf
        no_change = 0
        step = 0
        for _ in range(self.max_iter):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                batch = order[start : start + self.batch_size]
                _, gradient = parameter_gradient(
                    X[batch], y[batch], coefficients, self.dim, eig_idx, self.loss
                )
                step += 1
                momentum *= self.beta_1
                momentum += (1.0 - self.beta_1) * gradient
                variance *= self.beta_2
                variance += (1.0 - self.beta_2) * gradient**2
                coefficients -= (
                    self.learning_rate
                    * momentum
                    / (1.0 - self.beta_1**step)
                    / (np.sqrt(variance / (1.0 - self.beta_2**step)) + self.epsilon)
                )

            value = _objective(
                X, y, coefficients, self.dim, eig_idx, self.loss, self.batch_size
            )
            self.loss_curve_.append(value)
            no_change = 0 if value < best_loss - self.tol else no_change + 1
            best_loss = min(best_loss, value)
            if self.tol > 0.0 and no_change >= self.n_iter_no_change:
                break

        self.n_iter_ = len(self.loss_curve_)
        return SpectralModel(coefficients, self.dim, eig_idx)
