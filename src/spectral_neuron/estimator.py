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

from spectral_neuron.fitting import LBFGSFitter


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
        Mean training objective. Log loss accepts any two class labels.
    n_init : int, default=50
        Convex fits of random spectral features, counting opposite orientations
        separately. The lowest training-loss candidate initializes L-BFGS.
        An affine fallback is fitted once as needed; dim=1 uses only that fit.
    max_iter : int, default=300
        Maximum accepted L-BFGS steps. Zero returns the initialization.
    tol : float, default=1e-5
        Gradient infinity-norm threshold in normalized optimization coordinates;
        zero disables a positive threshold. This is a numerical stopping rule,
        not a general nonsmooth stationarity certificate.
    random_state : int or None, default=None
        Seed for the local initialization generator.

    Attributes
    ----------
    model_ : SpectralModel
        Fitted inference-only model in original input and target units.
    n_features_in_ : int
        Number of input features seen during fitting.
    initialization_losses_, initial_loss_ : list of float, float
        Candidate training losses and the selected initialization's loss.
    loss_curve_ : list of float
        Training loss after each accepted step, in original target units.
    n_iter_, n_evaluations_ : int
        Accepted steps and optimizer evaluations, excluding initialization and
        diagnostic evaluations.
    converged_, message_, gradient_norm_ : bool, str, float
        Whether the numerical gradient threshold was reached, the termination
        reason, and the final gradient infinity norm in optimization coordinates.
    feature_strengths_ : ndarray of shape (n_features_in_,)
        Spectral norms of the feature matrices. Each bounds raw output changes
        per unit change in that input feature, in the original units.
    loss_ : str
        Objective used by the most recent fit.
    classes_ : ndarray
        Ordered binary class labels; present only for log-loss fits.

    Notes
    -----
    Fitting standardizes inputs and regression targets, whitens the augmented
    design, and refines the best convex initialization with full-batch L-BFGS.
    All losses are exact. Both stages use training data only; the last accepted
    iterate is retained. Budget exhaustion or solver failure with a positive
    iteration budget raises ConvergenceWarning. See LBFGSFitter for details.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    n_init: int = 50
    max_iter: int = 300
    tol: float = 1e-5
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
        fitter = LBFGSFitter(**self.get_params(deep=False))
        self.model_ = fitter.fit(X, y.astype(np.float64, copy=False))
        for name in (
            "initialization_losses_", "initial_loss_", "loss_curve_", "n_iter_",
            "converged_", "message_", "n_evaluations_", "gradient_norm_",
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
