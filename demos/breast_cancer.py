# %% [markdown]
# # Breast cancer: feature selection
#
# Use the spectral-norm penalty to remove entire feature matrices, then refit
# without regularization on the selected features. Selection and scaling use
# training data only. Alpha is fixed for illustration, not tuned on the test set.
# The dataset's positive class (1) is benign.

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import RocCurveDisplay, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from spectral_neuron import SpectralNeuron

# %% Load the bundled dataset and hold out a stratified test split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=7
)

# %% Select features using exact zeros, without a numerical cutoff
selector = SpectralNeuron(dim=3, loss="log_loss", alpha=0.01, random_state=7)
selector.fit(X_train, y_train)
selected = selector.feature_strengths_ > 0
print(f"Selected {selected.sum()} of {len(selected)} features:")
print(", ".join(data.feature_names[selected]))

# %% Plot sensitivity bounds per training standard deviation
# Returned strengths use original units; rescale them for feature comparisons.
strengths = selector.feature_strengths_ * X_train.std(axis=0)
order = np.argsort(strengths)
fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
ax.barh(data.feature_names[order], strengths[order], color="tab:blue")
for label, kept in zip(ax.get_yticklabels(), selected[order]):
    label.set_color("tab:blue" if kept else "0.55")
ax.set(
    xlabel="Logit sensitivity bound per training standard deviation",
    title=f"Breast cancer: {selected.sum()}/{len(selected)} features retained (α = 0.01)",
)
plt.show()

# %% Refit selected features and compare with an unregularized all-feature fit
refit = SpectralNeuron(dim=3, loss="log_loss", random_state=7)
refit.fit(X_train[:, selected], y_train)
baseline = SpectralNeuron(dim=3, loss="log_loss", random_state=7)
baseline.fit(X_train, y_train)

# %% Report held-out performance; no choices are made using these scores
probabilities = {
    "Regularized selection": selector.predict_proba(X_test)[:, 1],
    "Selected features, refitted": refit.predict_proba(X_test[:, selected])[:, 1],
    "All features, unregularized": baseline.predict_proba(X_test)[:, 1],
}
print(f"\nTest set: {len(y_test)} samples")
print(f"{'Model':<30} {'Log loss':>10} {'ROC AUC':>10}")
for name, probability in probabilities.items():
    print(f"{name:<30} {log_loss(y_test, probability):>10.4f} "
          f"{roc_auc_score(y_test, probability):>10.4f}")

# %% Compare test ROC curves
fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
for name, probability in probabilities.items():
    RocCurveDisplay.from_predictions(y_test, probability, name=name, ax=ax)
ax.plot([0, 1], [0, 1], color="0.6", linestyle=":")
ax.set(title="Breast cancer: held-out classification", aspect="equal")
plt.show()
