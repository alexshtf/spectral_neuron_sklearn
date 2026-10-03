"""Spectral neurons with scikit-learn integration and proximal bundle fitting."""

from spectral_neuron.estimator import SpectralNeuron
from spectral_neuron.fitting import ProximalBundleFitter
from spectral_neuron.model import SpectralModel

__all__ = ["ProximalBundleFitter", "SpectralModel", "SpectralNeuron"]
