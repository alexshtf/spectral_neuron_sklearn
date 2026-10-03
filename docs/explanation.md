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

## Feature strengths

The fitted `SpectralNeuron.feature_strengths_` array has shape
`(n_features_in_,)` and contains the spectral norms $\|A_i\|_2$ of the learned
feature matrices, excluding $A_0$. Each value bounds the change in the raw
eigenvalue output when only that feature changes:

$$
|f(x + \delta e_i) - f(x)| \leq \|A_i\|_2 |\delta|.
$$

For `loss="log_loss"`, this is a bound on logits. Strengths are unnormalized
sensitivity bounds in the input units seen by the neuron; in a preprocessing
pipeline, they describe the transformed feature coordinates.

## Objective and scaling

`ProximalBundleFitter` standardizes each input column and, for regression, the
training targets. It minimizes

$$
F(A)=\frac1N\sum_{j=1}^N\ell\bigl(f(\tilde x_j),\tilde y_j\bigr)
+\alpha\sum_{i=1}^n\|A_i\|_{\mathrm{op}}.
$$

The bias matrix $A_0$ is unpenalized. `alpha=0` gives an unregularized fit;
positive values encourage small feature sensitivities and can set whole feature
matrices to zero. The penalty uses standardized coordinates. The returned
model and `feature_strengths_` use original input and target units.

Inputs are standardized column by column, without whitening: mixing feature
coordinates would couple their penalties and change the meaning of a zero
feature matrix. Classification targets remain binary labels encoded as 0 and 1.

## Initialization

For each of `n_init=50` draws, the fitter samples symmetric nonlinear directions
$T_i$. Feature directions have balanced extreme eigenvalues and joint scale

$$
\lambda_{\max}(T_i)=-\lambda_{\min}(T_i),\qquad
\sum_{i=1}^n\|T_i\|_{\mathrm{op}}=1.
$$

It then fits the affine coefficients in $A_i=p_iI+T_i$. Since identity shifts
preserve the selected eigenvalue index, the prediction is

$$
f(x)=p_0+\sum_{i=1}^nx_ip_i+
\lambda_k\!\left(T_0+\sum_{i=1}^nx_iT_i\right).
$$

The spectral term is fixed during this convex fit. Balanced extreme eigenvalues
also give $\|p_iI+T_i\|_{\mathrm{op}}=|p_i|+\|T_i\|_{\mathrm{op}}$, so the
regularizer reduces to an L1 penalty on the affine feature coefficients plus a
constant. Squared, absolute, and logistic loss all give convex subproblems.
The nonlinear scale stays fixed during initialization, avoiding an affine start
caused by a fitted scale of zero.

The candidate with the lowest penalized training objective starts refinement.
A separate convex affine fit provides a baseline; the fitter returns whichever
has the lower final training objective. With `dim=1`, initialization uses only
the affine fit. Initialization and model selection use training data only.

## Proximal bundle refinement

The fitter evaluates the exact loss and spectral derivatives on the full
training set. It keeps 20 recent loss values and gradients, together with the
current point, to build a local piecewise-linear model $m$ of the data loss.
Each trial approximately solves the convex problem

$$
\min_B\;m(B)+\alpha\sum_{i=1}^n\|B_i\|_{\mathrm{op}}
+\frac12\|B-A\|_H^2.
$$

The regularizer remains exact. Two BFGS curvature pairs define the positive
metric $H$. A small dual problem combines the stored gradients; its inner
computations use the operator-norm proximal map and a low-rank metric solve,
without evaluating the dataset's eigenvalues again. The proximal map can return
exactly zero feature matrices.

A trial is accepted when the actual objective reduction is at least one tenth
of the predicted reduction. Accepted trials weaken the quadratic penalty;
rejected trials strengthen it and still contribute information to the bundle.
There are no learning-rate, minibatch, smoothing, or patience settings.

Defaults are `n_init=50`, `max_iter=300`, and `tol=1e-6`. `max_iter` counts all
trials, including rejected ones. Zero keeps the best initialization or affine
baseline. `tol` controls a relative local model-decrease threshold; zero disables
the positive threshold. The convex subproblem's duality gap accounts for inner
solve accuracy when assessing that decrease.

## Fit diagnostics

| Attribute | Meaning |
| --- | --- |
| `initialization_objectives_` | Candidate objectives in standardized, penalized units |
| `initial_loss_`, `initial_objective_` | Loss and objective at the start of refinement |
| `loss_curve_`, `objective_curve_` | Loss and objective after each accepted trial |
| Fitter's `loss_`, `objective_` | Loss and objective of the returned model |
| `n_iter_`, `n_accepted_` | Trial count and accepted-trial count |
| `n_evaluations_` | Initial optimizer evaluation plus trials, excluding candidate fits and final diagnostics |
| `model_decrease_`, `duality_gap_` | Final local model decrease and inner accuracy bound |
| `used_affine_` | Whether the affine baseline supplied the returned model |
| `converged_`, `message_` | Stopping result and explanation |

Losses use original target units: mean squared error, mean absolute error, or
binary log loss. Objectives include the penalty and use standardized units.
Accepted objectives decrease; with regularization, the unpenalized loss can
increase. The curves describe nonlinear refinement, even when the affine
baseline supplies the returned model, and are empty when no trial is accepted.
`SpectralNeuron.loss_` identifies the loss by name; the numeric final loss is
available on the direct fitter.

`converged_` reports numerical completion. For bundle refinement, this means a
small local model decrease, not a general nonsmooth stationarity certificate.
Absolute-error kinks and eigenvalue collisions remain nonsmooth; the method has
no general convergence guarantee for this objective.
Budget exhaustion or numerical failure with a positive iteration budget raises
`ConvergenceWarning`. Unregularized logistic loss on separable data has no finite
minimizer, so a small training loss alone does not establish a useful fit.
