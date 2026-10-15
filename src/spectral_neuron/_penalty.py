"""Spectral-norm and anchored categorical-diameter penalties."""

from functools import cache
from itertools import groupby
from numbers import Integral

import numpy as np

from .model import _symmetric_matrices


@cache
def _packing(dim):
    i, j = np.tril_indices(dim)
    return i, j, np.where(i == j, 1.0, np.sqrt(2.0))


class SpectralPenalty:
    """Sum of feature norms and group diameters in isometric coordinates.

    Each group's first feature matrix is fixed to zero and omitted from
    ``free``. ``D`` maps the remaining coefficient matrices to pairwise
    differences and ungrouped matrices. Each ``blocks`` slice shares one
    nuclear-norm budget in the penalty's dual representation. Individual
    features are singleton blocks; consecutive equal-width blocks are batched
    while keeping their budgets independent.
    """

    def __init__(self, n_features, dim, alpha=0.0, onehot_groups=None):
        try:
            groups = [tuple(group) for group in (
                () if onehot_groups is None else onehot_groups
            )]
        except TypeError as error:
            raise ValueError(
                "onehot_groups must contain groups of feature indices"
            ) from error
        seen = set()
        for group in groups:
            if len(group) < 2 or any(
                isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                or not 1 <= index <= n_features for index in group
            ):
                raise ValueError(
                    "onehot_groups must contain at least two integer indices "
                    "in 1..n_features"
                )
            if len(set(group)) != len(group) or seen.intersection(group):
                raise ValueError(
                    "onehot_groups must be disjoint and contain distinct indices"
                )
            seen.update(group)

        anchors = {group[0] for group in groups}
        self.free = np.array([
            index for index in range(n_features + 1) if index not in anchors
        ])
        positions = {index: position for position, index in enumerate(self.free)}
        self.groups = [
            np.array([positions[index] for index in group[1:]]) for group in groups
        ]
        self.ungrouped = np.array([
            positions[index] for index in range(1, n_features + 1) if index not in seen
        ], dtype=int)
        self.shape = (len(self.free), dim * (dim + 1) // 2)
        self.dim, self.alpha = dim, alpha

        differences, inverses, self.blocks = [], [], []
        offset = 0
        for indices in [*self.groups, *self.ungrouped[:, None]]:
            k = len(indices) + 1
            i, j = np.triu_indices(k, 1)
            local = (np.eye(k)[i] - np.eye(k)[j])[:, 1:]
            # (L.T L)^-1 = (I + 11.T) / k for an anchored complete graph.
            inverse = (np.eye(k - 1) + np.ones((k - 1, k - 1))) @ local.T / k
            block = np.zeros((len(i), len(self.free)))
            block[:, indices] = local
            recovery = np.zeros_like(block.T)
            recovery[indices] = inverse
            differences.append(block)
            inverses.append(recovery)
            self.blocks.append(slice(offset, offset + len(i)))
            offset += len(i)
        self._block_runs = []
        for width, blocks in groupby(self.blocks, key=lambda block: block.stop - block.start):
            blocks = tuple(blocks)
            self._block_runs.append((blocks[0].start, blocks[-1].stop, width))
        self._block_starts = np.array([block.start for block in self.blocks], dtype=np.intp)
        self.D = np.vstack(differences)
        # Precompute the rank divisors used in each block's threshold.
        self._projection_ranks = np.arange(
            1, max(width for _, _, width in self._block_runs) * dim + 1, dtype=float,
        )
        self._inverse = np.hstack(inverses)
        self._operator = self.D if alpha else self.D[:0]
        self.norm_squared = max([
            1, *(len(group) + 1 for group in self.groups if len(group) > 1)
        ]) if alpha else 0

    def unweighted_value(self, coefficients):
        """Evaluate the mixed penalty before multiplying by alpha."""
        differences = self.D @ np.asarray(coefficients).reshape(self.shape)
        eigenvalues = np.linalg.eigvalsh(_symmetric_matrices(differences, self.dim))
        norms = np.max(np.abs(eigenvalues), axis=-1)
        # Accumulate the block maxima in block order.
        return float(sum(np.maximum.reduceat(norms, self._block_starts)))

    def __call__(self, coefficients):
        return self.alpha * self.unweighted_value(coefficients) if self.alpha else 0.0

    value = __call__

    def apply(self, coefficients):
        return self._operator @ np.asarray(coefficients).reshape(self.shape)

    def adjoint(self, differences):
        return (self._operator.T @ differences).ravel()

    def project(self, dual):
        """Return the projection onto block nuclear-norm balls and its normal.

        Reconstruct the normal by spectral clipping so inactive blocks remain
        exactly zero.
        """
        if not self.alpha:
            return np.zeros_like(dual), dual.copy()
        eigenvalues, vectors = np.linalg.eigh(_symmetric_matrices(dual, self.dim))
        caps = np.empty_like(eigenvalues[:, 0])
        for start, stop, width in self._block_runs:
            # Each row pools the eigenvalues of one block's matrices.
            ordered = np.abs(eigenvalues[start:stop]).reshape(-1, width * self.dim)
            ordered.sort(axis=-1)
            thresholds = (
                ordered[:, ::-1].cumsum(axis=-1) - self.alpha
            ) / self._projection_ranks[:ordered.shape[1]]
            caps[start:stop].reshape(-1, width)[:] = thresholds.max(axis=-1, keepdims=True)
        np.maximum(caps, 0.0, out=caps)
        clipped = np.clip(eigenvalues, -caps[:, None], caps[:, None])
        soft = np.sign(eigenvalues) * np.maximum(np.abs(eigenvalues) - caps[:, None], 0.0)
        i, j, factors = _packing(self.dim)

        def pack(values):
            matrices = (vectors * values[..., None, :]) @ vectors.swapaxes(-1, -2)
            return matrices[..., i, j] * factors

        return pack(soft), pack(clipped)

    def recover(self, normal, raw):
        """Recover primal matrices, preserving the unpenalized bias and zeros."""
        if not self.alpha:
            return np.asarray(raw).ravel().copy()
        coefficients = self._inverse @ normal
        coefficients[0] = np.asarray(raw).reshape(self.shape)[0]
        return coefficients.ravel()
