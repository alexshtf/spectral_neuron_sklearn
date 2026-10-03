"""Composite proximal bundle descent with a rank-four BFGS metric."""

from collections import deque

import numpy as np
from scipy.optimize import OptimizeResult, minimize as _minimize, root


def _inverse_product(vector, pairs):
    result, weights = vector.copy(), []
    for s, y, rho in reversed(pairs):
        weights.append(rho * (s @ result))
        result -= weights[-1] * y
    for (s, y, rho), weight in zip(pairs, reversed(weights)):
        result += s * (weight - rho * (y @ result))
    return result


def _metric(pairs, size):
    """Return U, d for the inverse metric H = I + U diag(d) U.T."""
    if not pairs:
        return np.empty((size, 0)), np.empty(0)
    span, _ = np.linalg.qr(np.column_stack([v for s, y, _ in pairs for v in (s, y)]))
    small = span.T @ np.column_stack([_inverse_product(v, pairs) for v in span.T])
    correction, rotation = np.linalg.eigh((small + small.T) / 2 - np.eye(span.shape[1]))
    if np.any(correction <= -1):
        raise np.linalg.LinAlgError("BFGS metric lost positive definiteness")
    return span @ rotation, correction


def _subproblem(centre, gradients, errors, damping, pairs, prox, penalty, weights):
    """Solve the simplex dual, using a root of dimension at most four per prox."""
    basis, correction = _metric(pairs, len(centre))
    latent = np.zeros(len(correction))
    centre_penalty = penalty(centre)

    def quadratic(step):
        return step @ step - np.sum(correction / (1 + correction) * (basis.T @ step)**2)

    def metric_prox(aggregate):
        nonlocal latent
        shifted = centre - (aggregate + basis @ (correction * (basis.T @ aggregate))) / damping

        def recover(a):
            argument = shifted - basis @ (correction * a) / damping
            # Ordinary prox recovery preserves exact zeros before root convergence.
            trial = prox(argument, 1 / damping)
            dual = (argument - trial) * damping
            residual = a - basis.T @ dual
            gap = np.sum(correction**2 / (1 + correction) * residual**2) / (2 * damping)
            return trial, residual, gap

        if len(correction):
            latent = root(lambda a: recover(a)[1], latent, method="hybr",
                          options={"xtol": 1e-10, "maxfev": 100}).x
        trial, _, gap = recover(latent)
        return trial, gap

    metric_norms = np.sum(gradients**2, axis=1)
    metric_norms += np.sum((gradients @ basis)**2 * correction, axis=1)
    scale = max(metric_norms.max() / damping, errors.max(), centre_penalty, 1e-12)

    def dual(weights):
        aggregate = weights @ gradients
        trial, _ = metric_prox(aggregate)
        step = trial - centre
        value = errors @ weights - aggregate @ step - penalty(trial) + centre_penalty
        value -= damping * quadratic(step) / 2
        return value / scale, (errors - gradients @ step) / scale

    fit = _minimize(
        dual, weights, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * len(errors),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones(len(w))},
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    weights = np.maximum(fit.x, 0.0)
    weights /= weights.sum()
    trial, prox_gap = metric_prox(weights @ gradients)
    step = trial - centre
    cuts = gradients @ step - errors
    predicted = centre_penalty - penalty(trial) - cuts.max() - damping * quadratic(step) / 2
    # The two gaps certify the convex subproblem, not the original objective.
    gap = max(float(cuts.max() - weights @ cuts + prox_gap), 0.0)
    return trial, float(predicted), weights, gap


def minimize(fun, prox, penalty, x, max_iter, tol, callback=None):
    """Minimize loss plus penalty; ``max_iter`` counts evaluated trial points.

    ``fun`` returns the data loss and gradient; ``prox`` and ``penalty`` include
    the regularization weight. The callback receives accepted (x, loss, total).
    Model tolerance is a numerical stopping rule, not a stationarity certificate.
    """
    x = x.copy()
    loss, gradient = fun(x)
    value = loss + penalty(x)
    bundle = deque([(x.copy(), loss, gradient.copy())], maxlen=20)
    pairs = deque(maxlen=2)
    weights = np.full(2, 0.5)
    damping, trials = 1.0, 0
    gap = decrease_bound = None
    success = False
    message = "Iteration budget exhausted."
    for _ in range(max_iter):
        points, losses, gradients = map(np.asarray, zip(*bundle, (x, loss, gradient)))
        distances = points - x
        errors = np.maximum(
            np.abs(losses - loss - np.sum(gradients * distances, axis=1)),
            np.sum(distances**2, axis=1),
        )
        try:
            trial, predicted, weights, gap = _subproblem(
                x, gradients, errors, damping, pairs, prox, penalty, weights,
            )
        except np.linalg.LinAlgError:
            message = "Convex subproblem lost numerical accuracy."
            break
        decrease_bound = predicted + gap
        if not np.isfinite([predicted, gap]).all():
            message = "Convex subproblem returned a nonfinite model bound."
            break
        if decrease_bound <= tol * max(1.0, abs(value)):
            success = True
            message = "Bundle model decrease below tolerance."
            break
        if predicted <= 0 or gap > 0.1 * predicted:
            message = "Convex subproblem accuracy limited progress."
            break
        trial_loss, trial_gradient = fun(trial)
        trials += 1
        trial_value = trial_loss + penalty(trial)
        if trial_value <= value - 0.1 * predicted:
            s, y = trial - x, trial_gradient - gradient
            curvature = s @ y
            if curvature > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                pairs.append((s, y, 1 / curvature))
            x, loss, gradient, value = trial, trial_loss, trial_gradient, trial_value
            damping /= 2
            if callback is not None:
                callback(x, loss, value)
        else:
            damping *= 2
        bundle.append((trial.copy(), trial_loss, trial_gradient.copy()))
        weights = np.r_[weights[:-1], 0.0, weights[-1]]
        if len(weights) > bundle.maxlen + 1:
            weights[-1] += weights[0]
            weights = weights[1:]
    return OptimizeResult(
        x=x, fun=value, loss=loss, jac=gradient, nit=trials, nfev=trials + 1,
        success=success, message=message, model_gap=gap, model_decrease_bound=decrease_bound,
    )
