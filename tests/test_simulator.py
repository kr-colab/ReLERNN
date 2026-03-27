"""
Unit tests for ReLERNN.simulator preprocessing methods.
"""

import copy
import numpy as np
import pytest

from ReLERNN.simulator import Simulator


def _make_simulator(**kwargs):
    defaults = dict(
        seed=42,
        N=10,
        phased=0,
        phaseError=0.0,
    )
    defaults.update(kwargs)
    return Simulator(**defaults)


# ---------------------------------------------------------------------------
# maskGenotypes
# ---------------------------------------------------------------------------

class TestMaskGenotypes:
    def test_removes_masked_sites(self):
        sim = _make_simulator()
        H = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        P = np.array([10, 20, 30, 40], dtype=np.float32)
        # Mask positions 15-25 (should remove position 20)
        rand_mask = [0.0, [[15, 25]]]
        H_out, P_out = sim.maskGenotypes(H, P, rand_mask)
        assert 20 not in P_out
        assert len(P_out) == 3
        assert H_out.shape[0] == 3

    def test_no_mask(self):
        sim = _make_simulator()
        H = np.array([[1, 0], [0, 1]], dtype=np.float32)
        P = np.array([10, 20], dtype=np.float32)
        rand_mask = [0.0, [[100, 200]]]
        H_out, P_out = sim.maskGenotypes(H, P, rand_mask)
        np.testing.assert_array_equal(P_out, P)
        assert H_out.shape == H.shape

    def test_all_masked(self):
        sim = _make_simulator()
        H = np.array([[1, 0], [0, 1]], dtype=np.float32)
        P = np.array([10, 20], dtype=np.float32)
        rand_mask = [1.0, [[0, 100]]]
        H_out, P_out = sim.maskGenotypes(H, P, rand_mask)
        assert len(P_out) == 0
        assert H_out.shape[0] == 0

    def test_multiple_mask_intervals(self):
        sim = _make_simulator()
        H = np.eye(5, dtype=np.float32)
        P = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        # Mask 15-25 and 35-45 -> removes positions 20 and 40
        rand_mask = [0.0, [[15, 25], [35, 45]]]
        H_out, P_out = sim.maskGenotypes(H, P, rand_mask)
        assert len(P_out) == 3
        np.testing.assert_array_equal(P_out, [10, 30, 50])


# ---------------------------------------------------------------------------
# phaseErrorer
# ---------------------------------------------------------------------------

class TestPhaseErrorer:
    def test_zero_rate(self):
        sim = _make_simulator()
        H = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        np.random.seed(42)
        result = sim.phaseErrorer(H, 0.0)
        np.testing.assert_array_equal(result, H)

    def test_shape_preserved(self):
        sim = _make_simulator()
        H = np.random.choice([0, 1], size=(100, 20)).astype(np.float32)
        result = sim.phaseErrorer(H, 0.5)
        assert result.shape == H.shape

    def test_original_unchanged(self):
        sim = _make_simulator()
        H = np.array([[1, 0], [0, 1]], dtype=np.float32)
        H_copy = H.copy()
        sim.phaseErrorer(H, 0.5)
        np.testing.assert_array_equal(H, H_copy)

    def test_full_rate_shuffles(self):
        sim = _make_simulator()
        np.random.seed(0)
        # Create a matrix where rows are very different
        H = np.zeros((50, 10), dtype=np.float32)
        H[:, 0] = 1.0  # only first column is 1
        result = sim.phaseErrorer(H, 1.0)
        # At rate=1.0, all sites get shuffled columns
        # The result should differ from original (statistically)
        assert not np.array_equal(result, H)
