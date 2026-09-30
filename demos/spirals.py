# %% [markdown]
# # Classifying two interleaving spirals
#
# Fit a spectral neuron with logistic loss to a positive and a negative
# spiral. Scikit-learn's DecisionBoundaryDisplay plots the decision boundary.

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

from spectral_neuron import SpectralNeuron

# %% Generate two spirals with a little noise
rng = np.random.default_rng(7)
theta = rng.uniform(0, 3 * np.pi, size=256)
radius = 0.1 + 0.9 * theta / (3 * np.pi)
positive = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
X = np.vstack((positive, -positive))
X += rng.normal(0, 0.015, size=X.shape)
y = np.repeat([1, -1], len(positive))
colors = {-1: "tab:blue", 1: "tab:orange"}

# %% Plot the classification data
fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
for label, color in colors.items():
    ax.scatter(*X[y == label].T, color=color, s=15, label=f"Class {label:+d}")
ax.set(xlabel="x₁", ylabel="x₂", title="Two interleaving spirals", aspect="equal")
ax.legend()
plt.show()

# %% Fit the classifier
neuron = SpectralNeuron(
    dim=9,
    loss="log_loss",
    max_iter=1000,
    random_state=7,
)
neuron.fit(X, y)

# %% Plot the decision boundary and the data
fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
display = DecisionBoundaryDisplay.from_estimator(
    neuron,
    X,
    response_method="predict_proba",
    grid_resolution=200,
    eps=0.1,
    levels=[0, 0.5, 1],
    colors=list(colors.values()),
    alpha=0.2,
    ax=ax,
)
display.plot(
    plot_method="contour", ax=ax, levels=[0.5], colors="black", linewidths=1
)
for label, color in colors.items():
    ax.scatter(*X[y == label].T, color=color, s=15, label=f"Class {label:+d}")
ax.set(
    xlabel="x₁", ylabel="x₂", title="Spectral neuron decision boundary", aspect="equal"
)
ax.legend()
plt.show()
