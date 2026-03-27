"""
Unit tests for ReLERNN.helpers
"""

import os
import tempfile
import numpy as np
import pytest

from ReLERNN.helpers import (
    assign_task,
    get_corrected_index,
    get_corrected,
    get_index,
    snps_per_win,
    find_win_size,
    force_win_size,
    maskStats,
    check_demHist,
    relu,
    mae,
    mse,
    unNormalize,
    sort_min_diff,
)


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------

class TestRelu:
    def test_positive(self):
        assert relu(5.0) == 5.0

    def test_negative(self):
        assert relu(-3.0) == 0

    def test_zero(self):
        assert relu(0) == 0


# ---------------------------------------------------------------------------
# mae / mse
# ---------------------------------------------------------------------------

class TestMaeMse:
    def test_mae_identical(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mae_known(self):
        assert mae([0, 0], [1, 3]) == pytest.approx(2.0)

    def test_mse_identical(self):
        assert mse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mse_known(self):
        assert mse([0, 0], [1, 3]) == pytest.approx(5.0)

    def test_mae_mse_length_mismatch(self):
        with pytest.raises(AssertionError):
            mae([1], [1, 2])
        with pytest.raises(AssertionError):
            mse([1], [1, 2])


# ---------------------------------------------------------------------------
# unNormalize
# ---------------------------------------------------------------------------

class TestUnNormalize:
    def test_round_trip(self):
        data = np.array([10.0, 20.0, 30.0])
        mean = np.mean(data)
        sd = np.std(data)
        zscored = (data - mean) / sd
        result = unNormalize(mean, sd, zscored.copy())
        np.testing.assert_array_almost_equal(result, data)

    def test_zero_sd(self):
        result = unNormalize(5.0, 0.0, np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [5.0, 5.0])


# ---------------------------------------------------------------------------
# snps_per_win
# ---------------------------------------------------------------------------

class TestSnpsPerWin:
    def test_uniform_positions(self):
        pos = np.array([5, 15, 25, 35, 45])
        result = snps_per_win(pos, 20)
        assert sum(result) == 5

    def test_clustered(self):
        pos = np.array([1, 2, 3, 100])
        result = snps_per_win(pos, 50)
        assert result[0] == 3
        assert sum(result) == 4

    def test_single_snp(self):
        pos = np.array([50])
        result = snps_per_win(pos, 100)
        assert len(result) == 1
        assert result[0] == 1


# ---------------------------------------------------------------------------
# get_index
# ---------------------------------------------------------------------------

class TestGetIndex:
    def test_contiguous_indices(self):
        pos = np.array([5, 15, 25, 35, 45])
        indices = get_index(pos, 20)
        # Indices should be contiguous
        for i in range(1, len(indices)):
            assert indices[i][0] == indices[i - 1][1]
        # Should cover all SNPs
        assert indices[0][0] == 0
        assert indices[-1][1] == 5

    def test_single_window(self):
        pos = np.array([1, 2, 3])
        indices = get_index(pos, 1000)
        assert len(indices) == 1
        assert indices[0] == [0, 3]


# ---------------------------------------------------------------------------
# find_win_size / force_win_size
# ---------------------------------------------------------------------------

class TestFindWinSize:
    def test_max_too_high(self):
        pos = np.array([1, 2, 3, 50, 51, 52])
        result = find_win_size(10, pos, winSizeMx=2)
        assert result == [-1]

    def test_max_too_low(self):
        pos = np.arange(1, 101)
        result = find_win_size(200, pos, winSizeMx=200)
        assert result == [1]

    def test_exact_match(self):
        pos = np.arange(1, 11)
        # Window size 10 -> 1 window with 10 SNPs
        result = find_win_size(10, pos, winSizeMx=10)
        assert len(result) == 5
        assert result[0] == 10  # winSize
        assert result[3] == 10  # max SNPs per window

    def test_force_win_size(self):
        pos = np.array([5, 15, 25, 35, 45])
        result = force_win_size(20, pos)
        assert len(result) == 5
        assert result[0] == 20  # winSize returned
        assert result[1] <= result[2] <= result[3]  # min <= mean <= max


# ---------------------------------------------------------------------------
# get_corrected_index
# ---------------------------------------------------------------------------

class TestGetCorrectedIndex:
    def test_exact_match(self):
        L = [1.0, 2.0, 3.0]
        idx, val = get_corrected_index(L, 2.0)
        assert idx == 1
        assert val == 2.0

    def test_nearest(self):
        L = [1.0, 5.0, 10.0]
        idx, val = get_corrected_index(L, 4.0)
        assert idx == 1
        assert val == 5.0

    def test_below_range(self):
        L = [10.0, 20.0]
        idx, val = get_corrected_index(L, 0.0)
        assert idx == 0
        assert val == 10.0

    def test_above_range(self):
        L = [10.0, 20.0]
        idx, val = get_corrected_index(L, 100.0)
        assert idx == 1
        assert val == 20.0


# ---------------------------------------------------------------------------
# get_corrected
# ---------------------------------------------------------------------------

class TestGetCorrected:
    def test_no_bias(self):
        bs = {
            "Q2": [1.0, 2.0, 3.0],
            "rho": [1.0, 2.0, 3.0],
            "CI95LO": [0.8, 1.8, 2.8],
            "CI95HI": [1.2, 2.2, 3.2],
        }
        cRATE, ciLO, ciHI = get_corrected(2.0, bs)
        assert cRATE == pytest.approx(2.0)
        assert ciLO < cRATE < ciHI

    def test_relu_clamp(self):
        bs = {
            "Q2": [0.1],
            "rho": [0.1],
            "CI95LO": [-10.0],
            "CI95HI": [0.2],
        }
        cRATE, ciLO, ciHI = get_corrected(0.1, bs)
        assert ciLO >= 0.0


# ---------------------------------------------------------------------------
# maskStats
# ---------------------------------------------------------------------------

class TestMaskStats:
    def _wins(self, chrom, start, length):
        return [f"{chrom}:0", start, length]

    def test_no_overlap_before(self):
        mask = {"chr1": [[0, 10]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == 0.0

    def test_no_overlap_after(self):
        mask = {"chr1": [[50, 60]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == 0.0

    def test_full_overlap(self):
        mask = {"chr1": [[0, 100]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == 1.0
        assert result[1] == [[0, 100]]

    def test_left_overlap(self):
        # mask [0, 25] overlaps window [20, 30) on the left
        mask = {"chr1": [[0, 25]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == pytest.approx(0.5)

    def test_right_overlap(self):
        # mask [25, 50] overlaps window [20, 30) on the right
        mask = {"chr1": [[25, 50]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == pytest.approx(0.5)

    def test_interior_mask(self):
        # mask [22, 28] is interior to window [20, 30)
        mask = {"chr1": [[22, 28]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == pytest.approx(0.6)

    def test_missing_chrom(self):
        mask = {"chr2": [[0, 100]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == 0.0

    def test_multiple_masks(self):
        mask = {"chr1": [[22, 24], [26, 28]]}
        result = maskStats(self._wins("chr1", 20, 10), 0, mask, 100)
        assert result[0] == pytest.approx(0.4)
        assert len(result[1]) == 2


# ---------------------------------------------------------------------------
# check_demHist
# ---------------------------------------------------------------------------

class TestCheckDemHist:
    def test_stairwayplot(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("mutation_per_site\t0.001\n")
            f.flush()
            assert check_demHist(f.name) == 1
        os.unlink(f.name)

    def test_smcpp(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("label,x,y\n")
            f.flush()
            assert check_demHist(f.name) == 2
        os.unlink(f.name)

    def test_msmc(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("time_index\tleft_time_boundary\n")
            f.flush()
            assert check_demHist(f.name) == 3
        os.unlink(f.name)

    def test_unknown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some random content\n")
            f.flush()
            assert check_demHist(f.name) == -9
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# assign_task
# ---------------------------------------------------------------------------

class TestAssignTask:
    def _collect(self, mpID, nProcs):
        from queue import Queue
        q = Queue()
        assign_task(mpID, q, nProcs)
        batches = []
        while not q.empty():
            batches.append(q.get())
        return batches

    def test_even_split(self):
        mpID = list(range(12))
        batches = self._collect(mpID, 4)
        all_ids = []
        for ids, _ in batches:
            all_ids.extend(ids)
        assert sorted(all_ids) == mpID
        assert len(batches) == 4

    def test_uneven_split(self):
        mpID = list(range(10))
        batches = self._collect(mpID, 3)
        all_ids = []
        for ids, _ in batches:
            all_ids.extend(ids)
        assert sorted(all_ids) == mpID
        assert len(batches) == 3

    def test_single_proc(self):
        mpID = list(range(5))
        batches = self._collect(mpID, 1)
        assert len(batches) == 1
        assert list(batches[0][0]) == mpID

    def test_more_procs_than_tasks(self):
        mpID = list(range(2))
        batches = self._collect(mpID, 4)
        all_ids = []
        for ids, _ in batches:
            all_ids.extend(ids)
        assert sorted(all_ids) == mpID


# ---------------------------------------------------------------------------
# sort_min_diff
# ---------------------------------------------------------------------------

class TestSortMinDiff:
    def test_permutation(self):
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        result = sort_min_diff(mat)
        assert result.shape == mat.shape
        # Result should be a permutation of rows
        assert set(map(tuple, result.tolist())) == set(map(tuple, mat.tolist()))

    def test_identical_rows(self):
        mat = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=float)
        result = sort_min_diff(mat)
        np.testing.assert_array_equal(result, mat)
