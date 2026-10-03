"""Composite proximal bundle descent with a rank-four BFGS metric."""

from collections import deque

import numpy as np
from scipy.optimize import OptimizeResult


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


def _simplex(vector):
    shifted = vector - vector.max()
    ordered = np.sort(shifted)[::-1]
    threshold = np.max((np.cumsum(ordered) - 1) / np.arange(1, len(vector) + 1))
    projected = np.maximum(shifted - threshold, 0)
    return projected / projected.sum()


def _subproblem(centre, gradients, errors, damping, pairs, penalty, weights, flow, goal):
    """Jointly project cut weights and penalty duals, certifying primal recovery."""
    # Parallel cuts differ only in height; only their lowest error matters.
    order = np.argsort(errors)
    _, first, inverse = np.unique(gradients[order], axis=0, return_index=True, return_inverse=True)
    indices = order[first]
    original_size = len(weights)
    weights = np.bincount(inverse, weights=weights[order])
    gradients, errors = gradients[indices], errors[indices]
    basis, correction = _metric(pairs, len(centre))

    def inverse_metric(vector):
        return vector + basis @ (correction * (basis.T @ vector))

    def quadratic(vector):
        coordinates = basis.T @ vector
        orthogonal = vector - basis @ coordinates
        return orthogonal @ orthogonal + np.sum(coordinates**2 / (1 + correction))

    projected = gradients @ basis
    gram = gradients @ gradients.T + (projected * correction) @ projected.T
    # Twice each block's curvature bounds the coupled dual Hessian.
    weight_step = .5 / max(np.linalg.eigvalsh(gram).max() / damping, 1e-10)
    flow_bound = np.max(1 + correction, initial=1) * penalty.norm_squared / damping
    flow_step = .5 / max(flow_bound, 1e-10)
    weights = weights.copy()
    flow = np.zeros_like(penalty.apply(centre)) if flow is None else flow.copy()
    accelerated_weights, accelerated_flow = weights.copy(), flow.copy()
    momentum = 1.0
    centre_penalty = penalty(centre)
    for _ in range(10000):
        aggregate = accelerated_weights @ gradients + penalty.adjoint(accelerated_flow)
        raw = centre - inverse_metric(aggregate) / damping
        next_weights = _simplex(
            accelerated_weights - weight_step * (errors - gradients @ (raw - centre))
        )
        next_flow, normal = penalty.project(accelerated_flow + flow_step * penalty.apply(raw))
        aggregate = next_weights @ gradients + penalty.adjoint(next_flow)
        raw = centre - inverse_metric(aggregate) / damping
        # Clipped spectra produce exact zeros; the gap includes recovery error.
        trial = penalty.recover(normal / flow_step, raw)
        step = trial - centre
        cuts = gradients @ step - errors
        trial_penalty = penalty(trial)
        predicted = centre_penalty - trial_penalty - cuts.max() - damping * quadratic(step) / 2
        gap = cuts.max() - next_weights @ cuts + trial_penalty
        gap -= np.sum(next_flow * penalty.apply(trial))
        gap = max(float(gap + damping * quadratic(trial - raw) / 2), 0.0)
        if (predicted + gap <= goal or predicted > 0 and gap <= .1 * predicted
                or not np.isfinite([predicted, gap]).all()):
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        restart = ((accelerated_weights - next_weights) @ (next_weights - weights) / weight_step
                   + np.sum((accelerated_flow - next_flow) * (next_flow - flow)) / flow_step)
        extrapolation = (momentum - 1) / next_momentum
        if restart > 0:
            next_momentum, extrapolation = 1.0, 0.0
        accelerated_weights = next_weights + extrapolation * (next_weights - weights)
        accelerated_flow = next_flow + extrapolation * (next_flow - flow)
        weights, flow, momentum = next_weights, next_flow, next_momentum
    weights = np.zeros(original_size)
    weights[indices] = next_weights
    return trial, float(predicted), weights, next_flow, gap


def minimize(fun, penalty, x, max_iter, tol, callback=None):
    """Minimize loss plus penalty; ``max_iter`` counts evaluated trial points.

    ``fun`` returns the data loss and gradient; ``penalty`` includes its weight.
    The callback receives accepted (x, loss, total). Model tolerance is a
    numerical stopping rule, not a stationarity certificate.
    """
    x = x.copy()
    loss, gradient = fun(x)
    value = loss + penalty(x)
    bundle = deque([(x.copy(), loss, gradient.copy())], maxlen=20)
    pairs = deque(maxlen=2)
    weights, flow = np.full(2, 0.5), None
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
        goal = tol * max(1.0, abs(value))
        try:
            trial, predicted, weights, flow, gap = _subproblem(
                x, gradients, errors, damping, pairs, penalty, weights, flow, goal,
            )
        except np.linalg.LinAlgError:
            message = "Convex subproblem lost numerical accuracy."
            break
        decrease_bound = predicted + gap
        if not np.isfinite([predicted, gap]).all():
            message = "Convex subproblem returned a nonfinite model bound."
            break
        if decrease_bound <= goal:
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
