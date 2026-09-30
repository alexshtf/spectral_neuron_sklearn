# %% [markdown]
# # Fitting a sine function
#
# Fit one spectral neuron to random samples of sin(3x) on [-1, 1], then
# compare its predictions with the true function on an evenly spaced grid.

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from spectral_neuron import SpectralNeuron

# %% Training data
rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, size=(256, 1))
y = np.sin(3 * X[:, 0])

# %% Fit the neuron
neuron = SpectralNeuron(dim=5, loss="squared_error", random_state=7)
neuron.fit(X, y)

# %% Predict on a dense grid
x_grid = np.linspace(-1, 1, 501)
true = np.sin(3 * x_grid)
fitted = neuron.predict(x_grid[:, None])

# %% Compare the true and fitted functions
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(x_grid, true, label="True: sin(3x)", linewidth=2)
ax.plot(x_grid, fitted, label="Spectral neuron", linestyle="--", linewidth=2)
ax.set(xlabel="x", ylabel="f(x)", title="Fitting a sine function")
ax.legend()
plt.show()
