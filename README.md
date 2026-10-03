# Spectral neuron component

A nonlinear model for regression and binary classification, with a scikit-learn
interface. Its prediction is one eigenvalue of a learned, input-dependent
symmetric matrix.

Uses NumPy, SciPy, and scikit-learn. Requires Python 3.12 or newer.

## Quick start

From a checkout of this repository, install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run this Python example in the project's environment (`uv run python`). It fits
a sine function from sampled input-output pairs:

```python
import numpy as np
from spectral_neuron import SpectralNeuron

rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(256, 1))
y = np.sin(3 * X[:, 0])

neuron = SpectralNeuron(random_state=7).fit(X, y)
print(neuron.predict([[0.2], [0.6]]))
```

![The true sine function and the fitted spectral neuron](docs/images/sine.png)

Use it in a preprocessing pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), SpectralNeuron(random_state=7))
model.fit(X, y)
print(model.predict([[0.2], [0.6]]))
```

## Learn with the demos

- [Sine](demos/sine.py): fit a function and compare it with the true curve.
- [Spirals](demos/spirals.py): classify two spirals and plot the decision boundary.
- [California housing](demos/california_housing.py): compare with linear
  regression using test R², RMSE, and residual histograms.

These scripts use `# %%` cells, so you can run them cell by cell or as ordinary
Python. For example:

```bash
uv run --group demos python demos/sine.py
```

Replace `sine.py` with another demo name to run it. The housing demo downloads
its dataset on the first run.

## API at a glance

Choose the task with `loss`:

| `loss` | Task | `predict(X)` returns |
| --- | --- | --- |
| `"squared_error"` (default) | Least-squares regression | Predicted values |
| `"absolute_error"` | Absolute-error regression | Predicted values |
| `"log_loss"` | Binary classification | Class labels |

For classification, `predict_proba(X)` returns probabilities and
`decision_function(X)` returns logits. `transform(X)` always returns the raw
eigenvalue as a single feature column, including for classification.

After fitting, `neuron.feature_strengths_` gives one sensitivity bound per input
feature for the raw eigenvalue output; see [feature strengths](docs/explanation.md#feature-strengths).

By default, `fitter="convex_lbfgs2"` selects the best of 50 convex fits of
random spectral features, then refines the matrices with full-batch L-BFGS and
a weak-Wolfe line search. It optimizes the exact requested loss, including
absolute error. Inputs and regression targets are scaled internally; predictions
and feature strengths use the original units. No learning rate is needed.

Defaults are `max_iter=300` and `tol=1e-5`. Set `max_iter=0` to retain the convex
initialization, or `tol=0` to disable the positive gradient threshold.
`initialization_losses_` and `initial_loss_` describe initialization;
`loss_curve_` records one training loss per accepted step. `converged_`,
`message_`, `n_evaluations_`, and `gradient_norm_` describe termination.
A numerical gradient criterion is not a general nonsmooth stationarity
certificate. Unsuccessful termination with a positive iteration budget raises
`ConvergenceWarning`.

The Adam alternatives remain available with `fitter="convex_adam"` and
`fitter="grouped_adam"`. Their learning-rate, batch, momentum, and patience
parameters do not apply to the default fitter. See
[initialization and training](docs/explanation.md#initialization), or
`help(SpectralNeuron)` for the full API.

## Fitting without the estimator

The default fitter is also available directly. Its `fit` method returns an
inference-only `SpectralModel`:

```python
from spectral_neuron import ConvexLBFGS2Fitter

fitter = ConvexLBFGS2Fitter(dim=5, random_state=7)
model = fitter.fit(X, y)
print(model([[0.2], [0.6]]))
print(fitter.loss_, fitter.message_)
```

For direct `loss="log_loss"` fitting, encode classes as 0 and 1; the model
returns logits. `SpectralNeuron` handles arbitrary binary class labels.

Other standalone fitters are retained:

- `ConvexAdamFitter` refines the best convex candidate with ordinary Adam.
- `ConvexAdam2Fitter` refines every candidate for exactly `n_iter=50` epochs at
  `learning_rate=1e-2`, then selects the lowest final training loss.
- `ConvexLBFGSFitter` uses SciPy's L-BFGS-B, with a smoothed approximation for
  absolute error.

See [the fitting methods](docs/explanation.md#full-batch-l-bfgs) for details.

## Understand the model

[How spectral neurons work](docs/explanation.md) explains eigenvalue selection,
initialization, and training. For the theory and experiments, see Alex Shtoff,
[*The Spectral Neuron*](https://arxiv.org/abs/2608.08003) (2026).

To run the tests: `uv run pytest`. Distributed under the [BSD 3-Clause license](LICENSE).
