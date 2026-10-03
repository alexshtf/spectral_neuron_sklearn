# %% [markdown]
# # Titanic: select numerical and categorical features
#
# Standardize numerical features and one-hot encode categorical features. The
# penalty selects each original column: a matrix norm for numerical features,
# a matrix diameter for categorical groups. Refit selected columns without the
# penalty. Preprocessing and selection use training data only; alpha is fixed
# for illustration, not tuned on the test set. The positive class is survival.

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import RocCurveDisplay, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spectral_neuron import SpectralNeuron

# %% Load and split the data (downloaded and cached on the first run)
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=7
)
numerical = ["age", "fare"]
categorical = ["sex", "pclass", "embarked"]
preprocessor = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numerical),
    ("cat", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False),
    ), categorical),
])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# %% Group indices follow the fitted encoder's output, using 1-based indexing
encoder = preprocessor.named_transformers_["cat"][-1]
widths = [1] * len(numerical) + [len(values) for values in encoder.categories_]
columns = np.split(np.arange(X_train.shape[1]), np.cumsum(widths)[:-1])
groups = [(column + 1).tolist() for column in columns[len(numerical):]]
selector = SpectralNeuron(
    dim=3, loss="log_loss", alpha=0.03, onehot_groups=groups, random_state=7
).fit(X_train, y_train)

# %% Measure whole-column strengths; zero diameter removes a whole category group
strengths = list(selector.feature_strengths_[:len(numerical)])
for group in groups:
    matrices = selector.model_.matrices[group]
    differences = matrices[:, None] - matrices[None, :]
    strengths.append(np.abs(np.linalg.eigvalsh(differences)).max())
strengths = np.asarray(strengths)
selected = strengths > 0
names = np.asarray(numerical + categorical)
print(f"Selected {selected.sum()} of {len(names)} original columns:")
print(", ".join(names[selected]))

order = np.argsort(strengths)
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.barh(names[order], strengths[order], color="tab:blue")
for label, retained in zip(ax.get_yticklabels(), selected[order]):
    label.set_color("tab:blue" if retained else "0.55")
ax.set(
    xlabel="Logit sensitivity bound: one numerical SD / a category swap",
    title=f"Titanic: {selected.sum()}/{len(names)} columns retained (α = {selector.alpha:g})",
)
plt.show()

# %% Keep every dummy of a selected group, including its zero reference matrix
keep = np.repeat(selected, widths)
remap = np.cumsum(keep)
refit_groups = [
    remap[np.asarray(group) - 1].tolist()
    for group, retained in zip(groups, selected[len(numerical):]) if retained
]
# At alpha=0, groups still fix a reference matrix to zero and leave indicators
# unscaled, removing redundant parameters without restricting one-hot fits.
refit = SpectralNeuron(
    dim=3, loss="log_loss", onehot_groups=refit_groups, max_iter=3000, random_state=7
).fit(X_train[:, keep], y_train)
baseline = SpectralNeuron(
    dim=3, loss="log_loss", onehot_groups=groups, max_iter=3000, random_state=7
).fit(X_train, y_train)

# %% Held-out performance after selection and unregularized refitting
probabilities = {
    "Regularized selection": selector.predict_proba(X_test)[:, 1],
    "Selected columns, refitted": refit.predict_proba(X_test[:, keep])[:, 1],
    "All columns, unregularized": baseline.predict_proba(X_test)[:, 1],
}
print(f"\nTest set: {len(y_test)} samples")
print(f"{'Model':<30} {'Log loss':>10} {'ROC AUC':>10}")
fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
for name, probability in probabilities.items():
    print(f"{name:<30} {log_loss(y_test, probability):>10.4f} "
          f"{roc_auc_score(y_test, probability):>10.4f}")
    RocCurveDisplay.from_predictions(y_test, probability, name=name, ax=ax)
ax.plot([0, 1], [0, 1], color="0.6", linestyle=":")
ax.set(title="Titanic: held-out classification", aspect="equal")
plt.show()
