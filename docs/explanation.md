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

## Initialization

The default `fitter="convex_lbfgs2"` and the other convex-initialized fitters
write each matrix as

$$
A_i=p_iI+\rho S_i,\qquad \operatorname{tr}(S_i)=0,
\qquad \sum_{i=0}^n\|S_i\|_F^2=1.
$$

It draws random shapes uniformly on this joint Frobenius unit sphere by
projecting Gaussian symmetric matrices to trace zero and normalizing them
together. For fixed shapes and nonnegative $\rho$, the prediction is

$$
f(x)=p_0+\sum_{i=1}^nx_ip_i
+\rho\lambda_k\!\left(S_0+\sum_{i=1}^nx_iS_i\right).
$$

Fitting $p$ and $\rho$ is convex: the spectral feature augments an affine
predictor. Initialization uses unregularized least squares for squared error,
a linear program for absolute error, and unregularized logistic regression
for log loss.
The lowest training-loss candidate initializes the matrices. This optimizes
the affine and scale parameters without a learning rate; the later matrix
optimization remains nonconvex.

`n_init=50` budgets candidate fits. An odd-dimensional middle eigenvalue allows
a signed amplitude, whose sign is absorbed into the shapes. Other indices
test both shape orientations, counting each fit separately; a negative fitted
amplitude uses the best affine fit, computed once as needed. With `dim=1`,
initialization is simply an affine fit. A zero amplitude can still yield a
degenerate affine starting point.

For absolute error, with augmented feature design $D$, initialization solves

$$
\max_{\lvert u_j\rvert\leq1}y^\top u
\quad\text{subject to}\quad D^\top u=0.
$$

This is the dual of least absolute deviations. The fitted coefficients are the
negative equality marginals returned by SciPy's `linprog(-y, ...)`; no residual
slack variables are needed.

## Full-batch L-BFGS

The default `ConvexLBFGS2Fitter` standardizes inputs and regression targets,
then selects the best convex initialization. An SVD of the augmented design
$D=[1,X]$ retains its numerical rank and gives $DT=Z$ with $Z^\top Z/N=I$.
Optimization uses packed symmetric coefficients $B$, for which

$$
\frac1N\sum_j\|\Delta A(x_j)\|_F^2=\|\Delta B\|_F^2.
$$

Whitening makes Euclidean parameter distance measure average matrix change on
the training data. Reconstruction $C=TB$ and inverse standardization restore
original input and target units. Rank-deficient designs use the minimum-norm
extension in standardized coordinates.

Full-batch L-BFGS uses 20 curvature pairs and an identity base metric. Each step
uses the analytic spectral derivative $vv^\top$ of a simple selected eigenvalue
and the exact squared, absolute, or logistic loss. A weak-Wolfe line search uses
Armijo constant $10^{-4}$, curvature constant $0.9$, and doubling or bisection.
Unlike strong Wolfe, weak Wolfe permits crossing an absolute-error kink without
requiring a smaller directional-derivative magnitude. The fixed identity metric
avoids scalar rescaling that can collapse near nonsmooth points; see
[Lewis and Overton](https://cs.nyu.edu/overton/papers/pdffiles/nsoquasi.pdf) and
[Asl and Overton](https://arxiv.org/abs/2006.11336).

Defaults are `n_init=50`, `max_iter=300`, and `tol=1e-5`. There are no
learning-rate, batch, momentum, or patience settings. `max_iter` counts accepted
steps, not line-search evaluations; zero returns the initialized model. `tol`
controls a numerical gradient threshold in normalized, whitened coordinates;
zero disables the positive threshold.

`initialization_losses_`, `initial_loss_`, `loss_curve_`, and the direct fitter's
`loss_` report exact training losses in original units. The curve has one value
per accepted step. `n_evaluations_` counts optimizer evaluations, excluding
initialization and diagnostic evaluations. It joins `n_iter_`, `gradient_norm_`,
`converged_`, and `message_` in distinguishing the gradient criterion, iteration
budget, and line-search failure. Unsuccessful termination with a positive budget
raises `ConvergenceWarning`. Initialization and refinement use training data only;
the last accepted iterate is returned as an inference-only `SpectralModel`.

This numerical gradient criterion is not a general nonsmooth stationarity
certificate. Absolute-error kinks and eigenvalue collisions remain nonsmooth,
and the method has no general convergence guarantee for this objective.

The earlier standalone `ConvexLBFGSFitter` retains SciPy's L-BFGS-B with the same
scaling and whitening. It uses exact squared and logistic losses, but replaces
absolute error by its Moreau envelope of width `2 * tol` in normalized target
units. Its loss discrepancy is at most `tol * target_scale`, with target scale
one for constant targets. This bounds approximation error, not optimization
error. `optimization_loss_` and `smoothing_error_bound_` describe that
approximation; its gradient and relative objective tolerances are `tol` and
`tol**2`.

## Adam refinement

`fitter="convex_adam"` selects `ConvexAdamFitter`. It refines the best convex
candidate with ordinary Adam, using a separate first and second moment for each
packed matrix coefficient. Standalone defaults match MLPRegressor's Adam:
`learning_rate=1e-3`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-8`,
`batch_size="auto"` (at most 200 samples), `max_iter=200`, and `tol=1e-4`.
Scale features and regression targets before fitting with Adam.

Training stops after `n_iter_no_change=10` epochs without sufficient improvement
over the best full training loss, including initialization, or at `max_iter`.
`converged_` records the patience criterion, not a certificate of an optimum.
Budget exhaustion raises `ConvergenceWarning`; `tol=0` disables patience and its
warning. `loss_curve_` records full losses after each epoch; the final iterate
is retained. When selected through `SpectralNeuron`, the estimator's explicit
parameter values apply, including its defaults `max_iter=300` and `tol=1e-5`.

## Refining every candidate

The standalone `ConvexAdam2Fitter` runs ordinary Adam from every convex candidate,
with fresh moments for each. Defaults are `n_init=50`, `n_iter=50` full epochs
per candidate, and `learning_rate=1e-2`. The lowest final full training loss
selects the returned model; there is no early stopping or checkpoint rollback.
Total work is about `n_init * n_iter` epochs; `dim=1` uses one affine start.

`initialization_losses_` and `final_losses_` record each candidate's losses;
`best_init_` is the winning zero-based index. `initial_loss_` and `loss_curve_`
describe that winner, and `n_iter_` counts its epochs. All direct fitters return
a `SpectralModel`, with logits for log loss.

## Original grouped Adam fitter

`fitter="grouped_adam"` retains the original near-affine initialization and
modified Adam. `feature_bound="auto"` sets the initialization scale from the
largest absolute training feature and records it in `feature_bound_`; these
apply only to this fitter. The selected eigenvalue initially has a gap on
the training inputs. Training can change that gap. See
[§7.2 of the paper](https://arxiv.org/html/2608.08003v2#S7.SS2).

Near this initialization, gradients that change affine behavior can be much
larger than those that develop nonlinearity. Grouped Adam gives them separate
scales.

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

For this fitter, `tol` instead measures the combined Frobenius norm of the
matrices' net changes over an epoch. Training stops after `n_iter_no_change`
consecutive changes at most `tol`, or at `max_iter`. Set `tol=0` to run the full
epoch budget. The final iterate is retained.
