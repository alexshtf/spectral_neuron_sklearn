# Spectral neuron component

A small scikit-learn estimator for supervised spectral neurons:

\[
f(x) = \lambda_k\left(A_0 + \sum_{i=1}^n x_i A_i\right),
\]

where the learned coefficient matrices are symmetric and `k` selects an
eigenvalue in ascending order. NumPy handles the spectral computations, SciPy
provides the sigmoid, and scikit-learn supplies estimator conventions and
validation. Analytical gradients and Adam updates with grouped gradient scales
are implemented internally.

## Installation

For development, use [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The package requires Python 3.12 or newer. Its dependency floors follow
[Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) as of
September 30, 2026: NumPy 2.2, SciPy 1.15, and scikit-learn 1.6.

## Fit and transform

```python
import numpy as np
from spectral_neuron import SpectralNeuron

rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(256, 1))
y = np.sin(3 * X[:, 0])

neuron = SpectralNeuron(dim=5, loss="squared_error", random_state=7)
neuron.fit(X, y)

prediction = neuron.predict(X)    # (256,), regression values
feature = neuron.transform(X)    # (256, 1), raw eigenvalue outputs
```

`fit(X, y)` returns the estimator itself. Inputs have shape
`(n_samples, n_features)` and targets `(n_samples,)`. A univariate input still
needs a feature axis, for example `x[:, None]`. Standard scikit-learn methods
provide `fit_transform`, `get_params`, `set_params`, cloning, feature-name
tracking, and transformer output support. The estimator works directly in
scikit-learn pipelines and model selection.

The default eigenvalue index is `dim // 2`. Set `eig_idx=0` for the smallest
eigenvalue (a concave function) or `eig_idx=dim - 1` for the largest (a convex
function). `dim=1` gives a linear model. The initial extraction supports dense,
unconstrained coefficient matrices.

## Demos

The [`sine` demo](demos/sine.py) expands the example above and plots the true
and fitted functions. The [`spirals` demo](demos/spirals.py) classifies two
interleaved spirals as positive and negative, plotting the data and decision
boundary with scikit-learn's `DecisionBoundaryDisplay`.

The [`California housing` demo](demos/california_housing.py) compares
`StandardScaler` pipelines with `SpectralNeuron` and `LinearRegression` on a
shared held-out split, reporting R² and RMSE and plotting residual histograms.
Errors are shown in dollars. `TransformedTargetRegressor` handles target scaling
for the neuron.
The first run downloads the dataset through scikit-learn.

Scripts in `demos/` use `# %%` cells for interactive execution and also run as
ordinary Python. Install the plotting and kernel dependencies with the `demos`
group:

```bash
uv sync --group demos
uv run --group demos python demos/sine.py
uv run --group demos python demos/spirals.py
uv run --group demos python demos/california_housing.py
```

## Objectives and prediction

The `loss` parameter selects a mean objective over the training samples:

| `loss` | Objective | `predict(X)` |
| --- | --- | --- |
| `"squared_error"` | `(f(x) - y)**2` | Scalar values, `(n_samples,)` |
| `"absolute_error"` | `abs(f(x) - y)` | Scalar values, `(n_samples,)` |
| `"log_loss"` | Binary cross-entropy from logits | Binary labels, `(n_samples,)` |

Absolute error uses a zero subgradient when the residual is exactly zero.
Logistic fitting accepts any two class labels, including strings, and treats
the selected eigenvalue as a logit. It uses stable log-domain loss and
`scipy.special.expit` for the sigmoid.

```python
binary_y = (X[:, 0] > 0).astype(int)
classifier = SpectralNeuron(dim=3, loss="log_loss", random_state=7).fit(X, binary_y)

labels = classifier.predict(X)           # classes_[1] when the logit is >= 0
probabilities = classifier.predict_proba(X)  # (n_samples, 2), classes_ order
logits = classifier.decision_function(X)  # (n_samples,)
features = classifier.transform(X)       # (n_samples, 1): raw logits
```

`predict_proba` and `decision_function` require a log-loss fit. Transformation
always returns raw eigenvalue outputs, regardless of the fitting objective.
Squared and absolute error identify a regressor with R² `score`; log loss
identifies a binary classifier with accuracy `score`.

## Fitting and inference

The component delegates training to an internal `AdamFitter` and retains only
the resulting `SpectralModel` and training summaries. Adam moment estimates
and training inputs are not stored on the component or inference model.

```python
inference_model = neuron.model_
raw = inference_model(X)     # (n_samples,)
matrices = inference_model.matrices  # (n_features + 1, dim, dim)
```

The inference model uses batched `numpy.linalg.eigvalsh` and accepts any
feature-last input shape `(..., n_features)`, preserving the leading shape.
It owns a copy of its fitted, read-only packed coefficients. It can be used
and serialized independently of the component.

Training uses `numpy.linalg.eigh` to obtain the selected eigenvector `v` and
the eigenvalue derivative `v @ v.T`. Matrices use isometric lower-triangular
coordinates: diagonal entries retain their scale and off-diagonal entries
are divided by `sqrt(2)` when embedded. Parameter gradients and updates are
vectorized across each batch. The paper's gapped base-matrix initialization
and jittered diagonal feature initialization are preserved, using a local
random generator.

Fitting first expresses all coefficient matrices in an eigenbasis of the
initial `A_0`, preserving every prediction and eigengap. The selected eigenvector
then occupies coordinate `k`. Because the initial feature matrices are close to
multiples of the identity, this direction approximately determines the selected
eigenvalue across inputs. Its diagonal entries control the nearly affine
prediction; the off-diagonal entries in its row and column allow that direction
to mix with the others and develop nonlinear behavior.

Adam keeps ordinary signed-gradient momentum for every packed parameter, but
shares its squared-gradient scale within three groups per coefficient matrix:
the selected diagonal entry, its off-diagonal row and column, and the remaining
submatrix. Each scale is an exponential moving average of the mean squared
packed gradients in that group. This separates the large gradients of the
nearly affine prediction from the smaller gradients that develop nonlinearity,
and makes updates independent of the choice of basis within the remaining
subspace. The grouping is fixed at initialization; all entries remain trainable.
For `dim=1`, only the single diagonal group exists and this is ordinary Adam.

By default, `feature_bound="auto"` takes `R` from `max(abs(X))` at fit time,
after any pipeline preprocessing. For all-zero inputs, it uses `R=1.0`.
The resolved value is stored in `feature_bound_`; the constructor parameter
stays `"auto"`. Set `feature_bound=5.0` to reproduce the paper's initialization,
or supply another positive, finite maximum absolute feature value.

Diagonal jitter is sampled within `1 / (4 * n_features * R)`, as in the
[paper's initialization lemma](https://arxiv.org/html/2608.08003#S7.SS2.SSS3).
The selected eigenvalue starts with a gap of at least `1/2` throughout
`abs(x_i) <= R`. This bound concerns initialization; fitting can change the gap.
With `"auto"`, the guarantee covers the training inputs and any other inputs
within that same bound.

`max_iter` counts epochs; each epoch shuffles and visits all samples once.
`batch_size` caps update and evaluation batches. `learning_rate` controls the
step size, `beta_1` the signed-gradient momentum, `beta_2` the squared-gradient
averages, and `epsilon` the denominator offset. Training stops after
`n_iter_no_change` consecutive epochs whose parameter displacement is at most
`tol`. Displacement is the Euclidean norm of the packed coefficient change,
equivalently
`sqrt(sum(||A_j_after - A_j_before||_F**2))` across all coefficient matrices.
This is an absolute threshold in parameter units; it detects when the matrices
have nearly stopped moving. Use `tol=0` to run every epoch. The final iterate
is retained. No validation or test data are used implicitly.

Inspect `loss_curve_` for the full-data training objective after each epoch,
and `n_iter_` for the number of completed epochs. Integer `random_state` makes
initialization and shuffling reproducible without changing global NumPy RNG
state.

At an eigenvalue collision, the selected eigenvalue may not be differentiable;
the fitter uses the selected eigenvector returned by NumPy. Absolute error
is also nonsmooth. Adam does not guarantee a global optimum for these models.

## Development

```bash
uv run pytest
uv run python -m compileall src tests
uv build
```

The code was extracted and adapted from the dense neuron in
[`spectral_neuron_paper`](https://github.com/alexshtf/spectral_neuron_paper).
The paper's experiment code remains in that repository. This library is
distributed under the BSD 3-Clause license.
