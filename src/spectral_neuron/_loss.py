"""Mean training losses and derivatives with respect to predictions."""

import numpy as np
from scipy.special import expit


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
