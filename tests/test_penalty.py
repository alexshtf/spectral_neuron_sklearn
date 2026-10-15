"""Spectral penalty values, dual projection, and coefficient recovery."""

from itertools import combinations

import numpy as np
import pytest

from spectral_neuron._penalty import SpectralPenalty
from spectral_neuron.model import _symmetric_matrices


LAYOUTS = [
    pytest.param(5, [], id="individual"),
    pytest.param(3, [[2, 3, 1]], id="one-group"),
    pytest.param(11, [[3, 1, 2], [6, 4, 5], [9, 7, 8]], id="equal-groups"),
    pytest.param(16, [[2, 1], [5, 3, 4], [9, 6, 8, 7], [14, 10, 13, 11, 12]],
                 id="mixed-level-counts"),
    pytest.param(15, [[3, 1, 2], [6, 4, 5], [10, 7, 9, 8], [13, 11, 12]],
                 id="separated-widths"),
    pytest.param(4, [[4, 2], [3, 1]], id="two-level-groups"),
]


@pytest.mark.parametrize("n_features, groups", LAYOUTS)
@pytest.mark.parametrize("dim", [1, 3, 5])
@pytest.mark.parametrize("alpha", [.02, .7, 3.])
@pytest.mark.parametrize("budget_ratio", [0., .25, 1., 8.],
                         ids=["zero", "interior", "boundary", "exterior"])
def test_projection_satisfies_block_budgets_and_optimality(
    n_features, groups, dim, alpha, budget_ratio,
):
    penalty = SpectralPenalty(n_features, dim, alpha, groups)
    rng = np.random.default_rng(615)
    dual = rng.normal(size=(len(penalty.D), penalty.shape[1]))
    spectra = np.linalg.eigvalsh(_symmetric_matrices(dual, dim))
    for block in penalty.blocks:
        dual[block] *= budget_ratio * alpha / np.abs(spectra[block]).sum()

    original_dual = dual.copy()
    projected, normal = penalty.project(dual)
    np.testing.assert_array_equal(dual, original_dual, strict=True)

    projected_spectra = np.linalg.eigvalsh(_symmetric_matrices(projected, dim))
    normal_norms = np.abs(np.linalg.eigvalsh(_symmetric_matrices(normal, dim))).max(axis=-1)
    for block in penalty.blocks:
        scale = max(alpha, np.linalg.norm(dual[block]))
        tolerance = 128 * np.finfo(float).eps * dim * (block.stop - block.start)
        assert np.abs(projected_spectra[block]).sum() <= alpha + tolerance * scale
        np.testing.assert_allclose(projected[block] + normal[block], dual[block],
                                   rtol=tolerance, atol=tolerance * scale)
        # The normal supports the feasible set at the projected point.
        np.testing.assert_allclose(
            np.sum(projected[block] * normal[block]), alpha * normal_norms[block].max(),
            rtol=tolerance, atol=tolerance * scale**2,
        )
    if budget_ratio < 1:
        np.testing.assert_array_equal(normal, 0)
        raw = rng.normal(size=penalty.shape)
        recovered = penalty.recover(normal, raw).reshape(penalty.shape)
        np.testing.assert_array_equal(recovered[0], raw[0], strict=True)
        np.testing.assert_array_equal(recovered[1:], 0)


def test_projection_uses_one_budget_per_group_and_per_individual_feature():
    penalty = SpectralPenalty(11, 1, 2., [[1, 2, 3], [4, 5, 6], [7, 8]])
    # The first two groups each share a budget across three scalar differences.
    # The two-level group and the three individual features get one budget each.
    dual = np.array([4., -3., 0., .25, -.5, .75, -3., 0., 1., 4.])[:, None]
    projected, normal = penalty.project(dual)
    np.testing.assert_array_equal(
        projected[:, 0], [1.5, -.5, 0., .25, -.5, .75, -2., 0., 1., 2.],
    )
    np.testing.assert_array_equal(
        normal[:, 0], [2.5, -2.5, 0., 0., 0., 0., -1., 0., 0., 2.],
    )


@pytest.mark.parametrize("n_features, groups", LAYOUTS)
@pytest.mark.parametrize("dim", [1, 3, 5])
def test_penalty_value_is_the_sum_of_feature_norms_and_group_diameters(n_features, groups, dim):
    penalty = SpectralPenalty(n_features, dim, .7, groups)
    coefficients = np.random.default_rng(228).normal(size=penalty.shape)
    matrices = np.zeros((n_features + 1, dim, dim))
    matrices[penalty.free] = _symmetric_matrices(coefficients, dim)
    grouped = {index for group in groups for index in group}
    # Evaluate all pairwise distances directly from the feature matrices.
    expected = sum(max(np.linalg.norm(matrices[i] - matrices[j], ord=2)
                       for i, j in combinations(group, 2)) for group in groups)
    expected += sum(np.linalg.norm(matrices[index], ord=2)
                    for index in range(1, n_features + 1) if index not in grouped)
    assert penalty.unweighted_value(coefficients) == pytest.approx(expected, rel=1e-13)
    assert penalty(coefficients) == pytest.approx(.7 * expected, rel=1e-13)
    before = penalty(coefficients)
    coefficients[0] += 100
    assert penalty(coefficients) == before  # The bias is unpenalized.


@pytest.mark.parametrize("n_rows", [0, 3])
def test_zero_alpha_preserves_supplied_dual_and_raw_arrays_by_copy(n_rows):
    penalty = SpectralPenalty(3, 3, 0., [[2, 1, 3]])
    dual = np.random.default_rng(17).normal(size=(n_rows, 6))
    projected, normal = penalty.project(dual)
    np.testing.assert_array_equal(projected, 0)
    np.testing.assert_array_equal(normal, dual, strict=True)
    assert not np.shares_memory(normal, dual)
    raw = np.arange(np.prod(penalty.shape), dtype=float).reshape(penalty.shape)
    recovered = penalty.recover(normal, raw)
    np.testing.assert_array_equal(recovered, raw.ravel(), strict=True)
    assert not np.shares_memory(recovered, raw)
    assert penalty(raw) == 0.
