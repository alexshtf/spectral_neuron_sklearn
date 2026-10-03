"""The scikit-learn spectral neuron estimator."""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from spectral_neuron._fitting import AdamFitter
from spectral_neuron._lbfgs import ConvexLBFGS2Fitter
from spectral_neuron.fitting import ConvexAdamFitter


@dataclass(kw_only=True, eq=False, repr=False)
class SpectralNeuron(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Learn one selected eigenvalue of an affine symmetric matrix pencil.

    Parameters
    ----------
    dim : int, default=5
        Matrix size. A size of one gives a linear model.
    eig_idx : int or None, default=None
        Zero-based index in ascending eigenvalue order; None selects dim // 2.
    loss : {'squared_error', 'absolute_error', 'log_loss'}
        Mean training objective. Log loss accepts any two class labels.
    fitter : {'convex_lbfgs2', 'convex_adam', 'grouped_adam'}
        Default 'convex_lbfgs2' uses convex initialization and exact-loss,
        unscaled weak-Wolfe L-BFGS. The Adam alternatives remain available.
    n_init : int, default=50
        Convex candidate fits, counting opposite orientations separately.
        Used by the convex fitters; a required affine fallback is fitted once
        in addition to this budget. A size-one model uses an affine fit directly.
    feature_bound : float or 'auto', default='auto'
        Initialization bound for 'grouped_adam' only. 'auto' uses the largest
        absolute training feature, or 1.0 if all features are zero.
    learning_rate : float, default=1e-3
        Adam step size; ignored by L-BFGS.
    max_iter : int, default=300
        Maximum L-BFGS steps or Adam epochs. For L-BFGS, zero returns the
        initialized model without refinement.
    batch_size : int or 'auto', default='auto'
        Adam samples per update; 'auto' uses min(200, n_samples). L-BFGS is
        full-batch and ignores this parameter.
    tol : float, default=1e-5
        L-BFGS gradient infinity-norm threshold in normalized coordinates;
        zero disables a positive threshold. For 'convex_adam', required loss
        improvement; for 'grouped_adam', small epoch displacement in Frobenius
        norm. A numerical gradient threshold is not a nonsmooth certificate.
    n_iter_no_change : int, default=10
        Stop after this many consecutive epochs without sufficient improvement
        (or with small displacement for 'grouped_adam'). Set tol=0 to disable
        Adam stopping. Ignored by L-BFGS.
    random_state : int or None, default=None
        Seed for a local NumPy generator used by initialization and shuffling.
    beta_1, beta_2 : float, default=0.9, 0.999
        Adam moment decay rates; ignored by L-BFGS.
    epsilon : float, default=1e-8
        Adam denominator offset; ignored by L-BFGS.

    Attributes
    ----------
    model_ : SpectralModel
        Fitted inference-only model; independent of optimizer state.
    n_features_in_ : int
        Number of features seen during fitting.
    feature_bound_ : float
        Resolved initialization bound; present only for 'grouped_adam'.
    initialization_losses_ : list of float
        Training losses of convex candidates; present for the convex fitters.
    initial_loss_ : float
        Selected initialization's training loss; present for the convex fitters.
    converged_ : bool
        Whether a numerical tolerance stopped L-BFGS or patience stopped
        'convex_adam'; neither certifies an optimum. L-BFGS budget exhaustion
        or failure raises ConvergenceWarning for a positive iteration budget.
    message_, n_evaluations_, gradient_norm_ : str, int, float
        L-BFGS termination reason, solver evaluations, and final gradient
        infinity norm in normalized coordinates.
    feature_strengths_ : ndarray of shape (n_features_in_,)
        Spectral norms of the learned feature matrices, excluding the intercept.
        Each value bounds the change in raw output per unit change in that
        input feature. For log loss, the raw output is the logit. Values are
        unnormalized and use the feature units received by the estimator.
    n_iter_ : int
        Number of accepted L-BFGS steps or completed Adam epochs.
    loss_curve_ : list of float
        Full training-data objective after each accepted step or epoch,
        in original target units.
    loss_ : str
        Objective used by the most recent fit.
    classes_ : ndarray
        Ordered class labels for a fitted log-loss component.

    Notes
    -----
    Convex initialization fits affine coefficients and the amplitude of each
    random spectral feature. The default fitter standardizes features and
    regression targets internally, whitens the design, and refines all matrix
    entries with exact-loss L-BFGS. Both stages use training data only. The
    returned model uses original units and the last accepted iterate. Scale
    inputs and regression targets yourself when selecting an Adam fitter.

    The optional grouped fitter shares squared-gradient scales over three
    parts of each matrix in the initial A_0 eigenbasis: the selected diagonal
    entry, its off-diagonal row and column, and the remaining submatrix.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    fitter: str = "convex_lbfgs2"
    n_init: int = 50
    feature_bound: float | str = "auto"
    learning_rate: float = 1e-3
    max_iter: int = 300
    batch_size: int | str = "auto"
    tol: float = 1e-5
    n_iter_no_change: int = 10
    random_state: int | None = None
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

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
        parameters = self.get_params(deep=False)
        method = parameters.pop("fitter")
        feature_bound = parameters.pop("feature_bound")
        if method == "convex_lbfgs2":
            fitter = ConvexLBFGS2Fitter(**{
                field.name: parameters[field.name] for field in fields(ConvexLBFGS2Fitter)
            })
        elif method == "convex_adam":
            fitter = ConvexAdamFitter(**parameters)
        elif method == "grouped_adam":
            parameters.pop("n_init")
            if feature_bound == "auto":
                feature_bound = float(np.abs(X).max()) or 1.0
            if parameters["batch_size"] == "auto":
                parameters["batch_size"] = min(200, len(X))
            fitter = AdamFitter(feature_bound=feature_bound, **parameters)
        else:
            raise ValueError("fitter must be 'convex_lbfgs2', 'convex_adam' or 'grouped_adam'")
        self.model_ = fitter.fit(X, y.astype(np.float64, copy=False))
        if method == "grouped_adam":
            self.feature_bound_ = float(feature_bound)
        else:
            self.__dict__.pop("feature_bound_", None)
        for name in (
            "initialization_losses_", "initial_loss_", "converged_",
            "message_", "n_evaluations_", "gradient_norm_",
        ):
            if hasattr(fitter, name):
                setattr(self, name, getattr(fitter, name))
            else:
                self.__dict__.pop(name, None)
        self._n_features_out = 1
        self.n_iter_ = fitter.n_iter_
        self.loss_curve_ = fitter.loss_curve_
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
