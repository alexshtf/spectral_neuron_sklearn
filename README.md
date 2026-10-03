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

## Learn with the demos

- [Sine](demos/sine.py): fit a function and compare it with the true curve.
- [Spirals](demos/spirals.py): classify two spirals and plot the decision boundary.
- [California housing](demos/california_housing.py): compare with linear
  regression using test R², RMSE, and residual histograms.
- [Breast cancer](demos/breast_cancer.py): select features with spectral-norm
  regularization, refit without the penalty, and compare held-out classification.

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

Fitting selects the best of `n_init=50` convex affine fits of random spectral
features, then refines the matrices with a full-batch proximal bundle method.
All three losses are exact. Inputs and regression targets are standardized
internally; predictions and feature strengths use the original units. No
learning rate is needed.

`alpha=0` is unregularized. A positive `alpha` penalizes the sum of feature
matrix operator norms in standardized coordinates and can set entire feature
matrices to zero. For example, use `SpectralNeuron(alpha=0.01)`.

Defaults are `max_iter=300` and `tol=1e-6`. The iteration budget counts every
trial, including rejected steps; `max_iter=0` returns the better initialization
or affine baseline. The final model is always compared with that baseline using
the training objective.

`loss_curve_` records accepted training losses in original units;
`objective_curve_` records their standardized, penalized objectives.
`converged_`, `message_`, `model_decrease_`, and `duality_gap_` describe
termination. A small local model decrease is not a general nonsmooth
stationarity certificate. Budget exhaustion or numerical failure with a
positive iteration budget raises `ConvergenceWarning`.

See [initialization and training](docs/explanation.md#initialization), or
`help(SpectralNeuron)` for the full API.

## Fitting without the estimator

`ProximalBundleFitter` exposes the same fitting procedure directly. Its `fit` method
returns an inference-only `SpectralModel`:

```python
from spectral_neuron import ProximalBundleFitter

fitter = ProximalBundleFitter(dim=5, random_state=7)
model = fitter.fit(X, y)
print(model([[0.2], [0.6]]))
print(fitter.loss_, fitter.message_)
```

For direct `loss="log_loss"` fitting, encode classes as 0 and 1; the model
returns logits. `SpectralNeuron` handles arbitrary binary class labels.

## Understand the model

[How spectral neurons work](docs/explanation.md) explains eigenvalue selection,
initialization, and training. For the theory and experiments, see Alex Shtoff,
[*The Spectral Neuron*](https://arxiv.org/abs/2608.08003) (2026).

To run the tests: `uv run pytest`. Distributed under the [BSD 3-Clause license](LICENSE).
