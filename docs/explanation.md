# How spectral neurons work

For usage examples, start with the [README](../README.md) or the
[demos](../demos/). This page explains the model and its training choices.

## A matrix becomes a prediction

A spectral neuron learns symmetric matrices `A₀, A₁, …, Aₙ`. Given an input
with features `x₁, …, xₙ`, it forms a weighted sum and returns one eigenvalue:

$$
A(x) = A_0 + \sum_{i=1}^{n} x_i A_i,
\qquad f(x) = \lambda_k(A(x)).
$$

The matrix depends linearly on the input, but its eigenvalues can vary
nonlinearly. That is where the model's ability to fit curved functions comes
from. With `dim=1`, the matrices are scalars and the model is an ordinary
affine function: an intercept plus a weighted sum of features.

Eigenvalues are ordered from smallest to largest. By default, `eig_idx=dim // 2`
selects the middle one. The smallest eigenvalue gives a concave function of the
input; the largest gives a convex function. An interior eigenvalue need not
have either shape.

For regression, the eigenvalue is the predicted value. For binary
classification, it is a logit: applying the sigmoid gives the probability of
the second class in `classes_`.

## Why initialization separates the eigenvalues

For a simple eigenvalue with normalized eigenvector `v`, its derivative with
respect to the matrix is `v vᵀ`. If the selected eigenvalue collides with another,
that derivative may no longer be unique.

The paper's initialization separates the selected eigenvalue from the others.
It uses a base matrix with a gap around the selected eigenvalue and feature
matrices that are multiples of the identity plus small diagonal perturbations.
These perturbations introduce nonlinearity while keeping the initial gap open.

Their size depends on a bound `R` on the absolute feature values. The default,
`feature_bound="auto"`, uses the largest absolute training feature after any
pipeline preprocessing, with `R=1` for all-zero inputs. Each perturbation is
bounded by `1 / (4 n R)`, giving an initial gap of at least `1/2` whenever every
feature has magnitude at most `R`. This is an initialization guarantee;
training can change the gap.

## Why Adam uses three gradient scales

The initial model is nearly affine. Gradients that change its affine behavior
can be much larger than gradients that develop its nonlinear behavior. The
optimizer gives these different kinds of changes separate scales.

First, a change of basis makes the selected eigenvector of the initial `A₀`
one coordinate axis. This leaves all predictions unchanged. In these
coordinates, each matrix gradient has three parts:

1. The diagonal entry on that axis.
2. The off-diagonal entries in its row and column, which connect that direction
   to the others.
3. The remaining symmetric submatrix.

Adam keeps a moving average of the signed gradient for every parameter. To
scale the updates, it also keeps a moving average of squared gradients,
averaged separately within each of these three parts of each matrix.
The smaller gradients therefore have their own scales. Sharing a scale within
each part also makes the update independent of the basis chosen for the
remaining subspace.

The groups stay fixed throughout fitting, and every matrix entry remains
trainable. With `dim=1`, this reduces to ordinary scalar Adam. As with Adam
generally, convergence to a global optimum is not guaranteed.

## When training stops

A flat loss curve can hide continuing changes to the model. Stopping therefore
depends on parameter movement: the combined Frobenius norm of all matrix
changes over an epoch. Training stops when this is at most `tol` for
`n_iter_no_change` consecutive epochs, or when `max_iter` is reached.
Set `tol=0` to use the full epoch budget.

`loss_curve_` records the training loss after each epoch. The final iterate is
retained; the estimator does not create a validation split or select a checkpoint.

## Fitting and prediction are separate

The estimator delegates optimization to an internal fitter. Its `model_`
attribute is a `SpectralModel` containing only the fitted parameters, with no
optimizer state or training data. You can call it directly to obtain raw
eigenvalues and inspect its `matrices` property. See the
[model docstring](../src/spectral_neuron/model.py) for its shape conventions.
