"""The scikit-learn spectral neuron estimator."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from spectral_neuron.fitting import ProximalBundleFitter


@dataclass(kw_only=True, eq=False, repr=False)
class SpectralNeuron(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Learn one selected eigenvalue of an affine symmetric matrix pencil.

    Parameters
    ----------
    dim : int, default=5
        Matrix size. A size of one gives a linear model.
    eig_idx : int or None, default=None
        Zero-based eigenvalue index in ascending order; None selects dim // 2.
    loss : {'squared_error', 'absolute_error', 'log_loss'}
        Exact mean data loss. Log loss accepts any two class labels.
    alpha : float, default=0
        Penalty on the sum of feature matrix operator norms in standardized
        input/target coordinates. The bias is unpenalized. Positive values
        can remove entire feature matrices exactly.
    n_init : int, default=50
        Convex affine calibrations of fixed-norm nonlinear candidates. The
        lowest penalized objective initializes refinement. An affine baseline
        is fitted separately and compared at the end; for dim=1, initialization
        uses only that affine fit.
    max_iter : int, default=300
        Maximum trial steps, including rejected trials. Zero returns the better
        of the nonlinear initialization and the affine baseline.
    tol : float, default=1e-6
        Relative bundle-model decrease tolerance; zero disables a positive
        threshold. This is not a nonsmooth stationarity certificate.
    random_state : int or None, default=None
        Seed for the local initialization generator.

    Attributes
    ----------
    model_ : SpectralModel
        Fitted inference-only model in original input and target units.
    n_features_in_ : int
        Number of input features seen during fitting.
    initialization_objectives_ : list of float
        Candidate penalized objectives in standardized coordinates.
    initial_loss_, initial_objective_ : float
        Nonlinear refinement's initial data loss in original units and its
        penalized objective in standardized coordinates.
    loss_curve_, objective_curve_ : list of float
        Accepted nonlinear steps: original-unit data losses and standardized
        penalized objectives. Data loss need not decrease when alpha is positive.
    objective_ : float
        Returned model's standardized penalized training objective.
    used_affine_ : bool
        Whether the final affine baseline had a lower objective than refinement.
    n_iter_, n_accepted_, n_evaluations_ : int
        Trials, accepted steps, and optimizer loss/gradient evaluations. The
        latter excludes initialization and diagnostics and equals n_iter_ + 1.
    converged_, message_ : bool, str
        Whether the model-decrease tolerance was reached and why fitting stopped.
    model_decrease_, duality_gap_ : float or None
        Last model's optimal decrease upper bound and primal-dual gap; None
        when max_iter=0. These are model diagnostics, not spectral certificates.
    feature_strengths_ : ndarray of shape (n_features_in_,)
        Spectral norms of the feature matrices. Each bounds raw output changes
        per unit change in that input feature, in the original units.
    loss_ : str
        Objective used by the most recent fit.
    classes_ : ndarray
        Ordered binary class labels; present only for log-loss fits.

    Notes
    -----
    Fitting standardizes inputs and regression targets, then uses a full-batch
    proximal bundle with two BFGS curvature pairs and an exact spectral-norm
    prox. All stages use training data only. Budget exhaustion or solver failure
    with a positive iteration budget raises ConvergenceWarning. See
    ProximalBundleFitter for details.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    alpha: float = 0.0
    n_init: int = 50
    max_iter: int = 300
    tol: float = 1e-6
    random_state: int | None = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        if self.loss == "log_loss":
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags(multi_class=False)
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SpectralNeuron":
        """Fit to dense tabular inputs and one target per sample."""
        X, y = validate_data(
            self, X, y, dtype=np.float64, y_numeric=self.loss != "log_loss"
        )
        if self.loss == "log_loss":
            check_classification_targets(y)
            classes, y = np.unique(y, return_inverse=True)
            if len(classes) != 2:
                raise ValueError(
                    "Only binary classification is supported. "
                    f"Got {len(classes)} class(es)."
                )
        fitter = ProximalBundleFitter(**self.get_params(deep=False))
        self.model_ = fitter.fit(X, y.astype(np.float64, copy=False))
        for name in (
            "initialization_objectives_", "initial_loss_", "initial_objective_",
            "loss_curve_", "objective_curve_", "objective_", "used_affine_",
            "n_iter_", "n_accepted_", "n_evaluations_", "converged_", "message_",
            "model_decrease_", "duality_gap_",
        ):
            setattr(self, name, getattr(fitter, name))
        self._n_features_out = 1
        self.loss_ = self.loss
        if self.loss_ == "log_loss":
            self.classes_ = classes
        elif hasattr(self, "classes_"):
            del self.classes_
        return self

    @property
    def feature_strengths_(self) -> NDArray[np.float64]:
        """Return the spectral norm of each learned feature matrix."""
        check_is_fitted(self, "model_")
        eigenvalues = np.linalg.eigvalsh(self.model_.matrices[1:])
        return np.abs(eigenvalues).max(axis=-1)

    def _raw_prediction(self, X):
        check_is_fitted(self, "model_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self.model_._predict(X)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return raw eigenvalue outputs as a single feature column."""
        return self._raw_prediction(X)[:, None]

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return scalar regression values or binary labels for log-loss fits."""
        raw = self._raw_prediction(X)
        if self.loss_ == "log_loss":
            return self.classes_[(raw >= 0).astype(np.int64)]
        return raw

    @available_if(lambda self: self.loss == "log_loss")
    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return logits from a fitted log-loss component."""
        return self._raw_prediction(X)

    @available_if(lambda self: self.loss == "log_loss")
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return probability columns in classes_ order."""
        p = expit(self.decision_function(X))
        return np.column_stack((1 - p, p))

    def score(self, X, y, sample_weight=None):
        """Return accuracy for log loss, or R² for the regression losses."""
        prediction = self.predict(X)
        metric = accuracy_score if self.loss_ == "log_loss" else r2_score
        return metric(y, prediction, sample_weight=sample_weight)
