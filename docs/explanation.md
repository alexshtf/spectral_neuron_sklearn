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

`LBFGSFitter` initializes each matrix as

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

The fitter standardizes inputs and regression targets before initialization.
An SVD of the augmented design
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
