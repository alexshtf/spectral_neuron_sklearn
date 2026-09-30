# %% [markdown]
# # California housing regression
#
# Compare a spectral neuron with linear regression on the same held-out data.
# Each pipeline standardizes features using only the training split.
# The spectral neuron's target scaling is learned from the training split and
# reversed automatically for prediction. RMSE and residuals are in dollars.

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectral_neuron import SpectralNeuron

# %% Load and split the data (downloaded on the first run)
X, y = fetch_california_housing(return_X_y=True)
y = y * 100_000  # The loader expresses house values in $100,000 units.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# %% Fit both models
models = {
    "Spectral neuron": make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(
            regressor=SpectralNeuron(dim=11, random_state=7), transformer=StandardScaler()
        ),
    ),
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
}
for model in models.values():
    model.fit(X_train, y_train)

# %% Compare test R² and RMSE
predictions = {name: model.predict(X_test) for name, model in models.items()}
print(f"Test set: {len(y_test):,} samples")
print(f"{'Model':<20} {'R²':>8} {'RMSE ($)':>12}")
for name, prediction in predictions.items():
    r2 = r2_score(y_test, prediction)
    rmse = root_mean_squared_error(y_test, prediction)
    print(f"{name:<20} {r2:>8.3f} {rmse:>12,.0f}")

# %% Compare test residual distributions
residuals = np.column_stack([y_test - prediction for prediction in predictions.values()])
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
# Passing both columns together gives the histograms identical bin edges.
ax.hist(
    residuals,
    bins="fd",
    histtype="step",
    color=["tab:blue", "tab:orange"],
    linestyle=["-", "--"],
    linewidth=1.8,
    label=list(predictions),
)
ax.axvline(0, color="0.4", linewidth=1, linestyle=":")
# Keep the center readable while retaining large residuals on logarithmic tails.
ax.set_xscale("symlog", linthresh=100_000)
ax.set(
    xlabel="Residual: actual − predicted ($; symmetric log scale)",
    ylabel="Test samples",
    title="California housing: test residuals",
)
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.legend()
plt.show()
