# How spectral neurons work

A spectral neuron combines learned symmetric matrices using the input features
as weights, then returns one eigenvalue:

$$
A(x) = A_0 + \sum_{i=1}^{n} x_i A_i,
\qquad f(x) = \lambda_k(A(x)).
$$

Taking an eigenvalue introduces nonlinearity. With `dim=1`, this reduces to an
intercept plus a weighted sum of features. The default selects the middle
eigenvalue; choosing the smallest or largest gives a concave or convex function.
See [*The Spectral Neuron*, §4](https://arxiv.org/html/2608.08003v2#S4) for the theory,
and the [README](../README.md) for usage.

## Initialization

By default, the model starts nearly affine, with its selected eigenvalue
separated from the others on the training inputs. `feature_bound="auto"` sets
the initialization scale from the largest absolute training feature, after
preprocessing. Training can change the initial eigenvalue gap. The construction
and its guarantees are in [§7.2 of the paper](https://arxiv.org/html/2608.08003v2#S7.SS2).

## Our Adam variant

This library uses a modification of Adam that is not described in the paper.
Near initialization, gradients that change the model's affine behavior can be
much larger than those that develop its nonlinear behavior. We give these
different kinds of changes separate scales.

At the start of fitting, we express every matrix in a basis where the selected
eigenvector of the initial `A₀` is one coordinate axis. This preserves all
predictions. For each learned matrix separately, we split the loss gradient
with respect to that matrix into three parts:

1. The diagonal entry on that axis.
2. The off-diagonal entries in its row and column, which connect that direction
   to the others.
3. The remaining symmetric submatrix.

As in ordinary Adam, each parameter keeps its own moving average of signed
gradients: its momentum. We change the scale estimate: each part shares one
moving average of its mean squared-gradient size. Each parameter's update uses
its own momentum divided by the square root of its part's scale estimate,
with Adam's usual bias corrections, learning rate, and epsilon.

For precision, a part's mean square is its squared Frobenius norm divided by
its number of independent parameters. For `d = dim`, the three counts are
`1`, `d − 1`, and `d(d − 1)/2`. The Frobenius norm counts both symmetric copies
of off-diagonal entries.

This gives smaller gradients their own scale and makes updates independent of
the basis chosen for the remaining subspace. The basis and groups stay fixed;
every entry remains trainable. No hyperparameters are added. With `dim=1`, only
the first part exists and the update is ordinary scalar Adam.

## Stopping and prediction

Training stops when the combined Frobenius norm of the matrices' net changes
over an epoch is at most `tol` for `n_iter_no_change` consecutive epochs, or at
`max_iter`. Set `tol=0` to run the full epoch budget. `loss_curve_` records the
training loss; the final iterate is retained.

The fitted `model_` contains the parameters needed for prediction, without
optimizer state. See the [API docstrings](../src/spectral_neuron/estimator.py)
for parameter details.
