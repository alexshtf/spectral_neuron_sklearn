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

from spectral_neuron._fitting import AdamFitter


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
    feature_bound : float or 'auto', default='auto'
        Positive bound R on absolute feature values for initialization. Diagonal
        jitter is sampled from [-1 / (4 * n_features * R), 1 / (4 * n_features * R)].
        'auto' uses the absolute maximum of the training features, or 1.0 if all
        features are zero. The paper's initial eigengap guarantee applies when
        each feature's magnitude is at most R.
    learning_rate : float, default=0.01
        Adam step size.
    max_iter : int, default=200
        Maximum number of epochs, each visiting every sample once.
    batch_size : int, default=128
        Maximum number of samples per update and evaluation batch.
    tol : float, default=1e-6
        Maximum absolute parameter displacement per epoch considered small.
        Measured as the Euclidean norm of all packed coefficient changes,
        equivalently the combined Frobenius norm of the matrix changes.
    n_iter_no_change : int, default=10
        Stop after this many consecutive epochs with displacement at most tol.
        Set tol=0 to disable this stopping criterion.
    random_state : int or None, default=None
        Seed for a local NumPy generator used by initialization and shuffling.
    beta_1, beta_2 : float, default=0.9, 0.999
        Decay rates of per-parameter signed-gradient momentum and per-group
        mean squared-gradient estimates, respectively.
    epsilon : float, default=1e-8
        Adam denominator offset.

    Attributes
    ----------
    model_ : SpectralModel
        Fitted inference-only model; independent of optimizer state.
    n_features_in_ : int
        Number of features seen during fitting.
    feature_bound_ : float
        Resolved feature bound used by the most recent fit.
    n_iter_ : int
        Number of completed epochs.
    loss_curve_ : list of float
        Full training-data objective after each completed epoch.
    loss_ : str
        Objective used by the most recent fit.
    classes_ : ndarray
        Ordered class labels for a fitted log-loss component.

    Notes
    -----
    Fitting uses Adam with three squared-gradient scales per coefficient
    matrix. In an eigenbasis of the initial A_0, the groups are the selected
    diagonal entry, its off-diagonal row and column, and the remaining
    submatrix. Separate scales prevent the dominant gradients of the initially
    nearly affine prediction from setting the scale for smaller gradients
    that develop nonlinearity. Each parameter retains its own signed-gradient
    momentum. The basis and groups stay fixed during fitting; all matrix
    entries remain trainable. The initial basis change preserves predictions
    and eigengaps. For dim=1, the update is ordinary scalar Adam.
    """

    dim: int = 5
    eig_idx: int | None = None
    loss: str = "squared_error"
    feature_bound: float | str = "auto"
    learning_rate: float = 0.01
    max_iter: int = 200
    batch_size: int = 128
    tol: float = 1e-6
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
        feature_bound = self.feature_bound
        if isinstance(feature_bound, str) and feature_bound == "auto":
            feature_bound = float(max(-X.min(), X.max())) or 1.0
        fitter = AdamFitter(
            **(self.get_params(deep=False) | {"feature_bound": feature_bound})
        )
        self.model_ = fitter.fit(X, y.astype(np.float64))
        self.feature_bound_ = float(feature_bound)
        self._n_features_out = 1
        self.n_iter_ = fitter.n_iter_
        self.loss_curve_ = fitter.loss_curve_
        self.loss_ = self.loss
        if self.loss_ == "log_loss":
            self.classes_ = classes
        elif hasattr(self, "classes_"):
            del self.classes_
        return self

    def _raw_prediction(self, X):
        check_is_fitted(self, "model_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self.model_(X)

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
