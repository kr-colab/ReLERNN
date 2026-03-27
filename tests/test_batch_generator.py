"""
Unit tests for ReLERNN.sequenceBatchGenerator preprocessing methods.
"""

import copy
import pickle
import tempfile
import os
import numpy as np
import pytest

from ReLERNN.sequenceBatchGenerator import SequenceBatchGenerator


def _make_generator(tmpdir, n=100, nSamps=10, targetNorm='zscore',
                    maxLen=50, frameWidth=0, seqD=None, maf=0.0):
    """Create a minimal SequenceBatchGenerator with synthetic data."""
    # Create fake simulation outputs
    for i in range(n):
        nSNPs = np.random.randint(10, 40)
        H = np.random.choice([0, 1], size=(nSNPs, nSamps)).astype(np.float32)
        P = np.sort(np.random.uniform(0, 1, nSNPs)).astype(np.float32)
        np.save(os.path.join(tmpdir, f"{i}_haps.npy"), H)
        np.save(os.path.join(tmpdir, f"{i}_pos.npy"), P)

    info = {
        "rho": np.random.uniform(0.0001, 0.01, n).astype(np.float32),
        "segSites": np.random.randint(10, 40, n),
        "numReps": n,
        "ChromosomeLength": 1e5,
    }
    pickle.dump(info, open(os.path.join(tmpdir, "info.p"), "wb"))

    gen = SequenceBatchGenerator(
        treesDirectory=tmpdir,
        targetNormalization=targetNorm,
        batchSize=16,
        maxLen=maxLen,
        frameWidth=frameWidth,
        shuffleInds=False,
        sortInds=False,
        center=False,
        ancVal=-1,
        padVal=0,
        derVal=1,
        realLinePos=True,
        posPadVal=0,
        seqD=seqD,
        maf=maf,
        seed=42,
    )
    return gen


# ---------------------------------------------------------------------------
# pad_HapsPos
# ---------------------------------------------------------------------------

class TestPadHapsPos:
    def test_output_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, maxLen=50)
            haps = [np.random.randn(20, 10).astype(np.float32),
                    np.random.randn(30, 10).astype(np.float32)]
            pos = [np.linspace(0, 1, 20).astype(np.float32),
                   np.linspace(0, 1, 30).astype(np.float32)]
            h, p = gen.pad_HapsPos(haps, pos, maxSNPs=50)
            assert h.shape == (2, 50, 10)
            assert p.shape == (2, 50)

    def test_padding_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, maxLen=50)
            haps = [np.zeros((10, 5), dtype=np.float32)]
            pos = [np.ones(10, dtype=np.float32)]
            h, p = gen.pad_HapsPos(haps, pos, maxSNPs=20)
            # Padded region should be 2.0 for haps, -1.0 for pos
            assert h[0, 10, 0] == 2.0
            assert p[0, 10] == -1.0
            # Original region preserved
            assert h[0, 0, 0] == 0.0
            assert p[0, 0] == 1.0

    def test_center_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, maxLen=50)
            haps = [np.ones((10, 5), dtype=np.float32)]
            pos = [np.ones(10, dtype=np.float32) * 0.5]
            h, p = gen.pad_HapsPos(haps, pos, maxSNPs=20, center=True)
            # 10 padding total -> 5 before, 5 after
            assert h[0, 0, 0] == 2.0  # padding before
            assert h[0, 5, 0] == 1.0  # real data starts at index 5
            assert h[0, 15, 0] == 2.0  # padding after

    def test_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, maxLen=50)
            haps = [np.ones((30, 5), dtype=np.float32)]
            pos = [np.linspace(0, 1, 30).astype(np.float32)]
            h, p = gen.pad_HapsPos(haps, pos, maxSNPs=20)
            assert h.shape == (1, 20, 5)
            assert p.shape == (1, 20)

    def test_framewidth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, maxLen=50)
            haps = [np.ones((10, 5), dtype=np.float32)]
            pos = [np.ones(10, dtype=np.float32)]
            h, p = gen.pad_HapsPos(haps, pos, maxSNPs=10, frameWidth=3)
            # 10 + 2*3 = 16 for SNP dim, 5 + 2*3 = 11 for sample dim
            assert h.shape == (1, 16, 11)
            assert p.shape == (1, 16)


# ---------------------------------------------------------------------------
# normalizeTargets
# ---------------------------------------------------------------------------

class TestNormalizeTargets:
    def test_zscore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, targetNorm='zscore')
            normed = gen.normalizeTargets()
            assert normed.mean() == pytest.approx(0.0, abs=1e-5)
            assert normed.std() == pytest.approx(1.0, abs=1e-2)

    def test_divstd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, targetNorm='divstd')
            normed = gen.normalizeTargets()
            assert normed.std() == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# normalizeAlleleFqs
# ---------------------------------------------------------------------------

class TestNormalizeAlleleFqs:
    def test_zscore_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, targetNorm='zscore')
            fqs = [np.random.uniform(0, 1, 20).astype(np.float32),
                   np.random.uniform(0, 1, 30).astype(np.float32)]
            result = gen.normalizeAlleleFqs(fqs)
            # After zscore, the pooled values should have mean ~0
            pooled = np.concatenate([r.flatten() for r in result])
            assert pooled.mean() == pytest.approx(0.0, abs=1e-5)

    def test_constant_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, targetNorm='zscore')
            fqs = [np.ones(10, dtype=np.float32) * 0.5]
            result = gen.normalizeAlleleFqs(fqs)
            # std=0 -> divide by zero guarded -> should be zeros
            np.testing.assert_array_almost_equal(result[0], 0.0)


# ---------------------------------------------------------------------------
# normalizeTargetsBinaryClass
# ---------------------------------------------------------------------------

class TestNormalizeTargetsBinaryClass:
    def test_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 50
            for i in range(n):
                nSNPs = 10
                H = np.zeros((nSNPs, 5), dtype=np.float32)
                P = np.linspace(0, 1, nSNPs).astype(np.float32)
                np.save(os.path.join(tmpdir, f"{i}_haps.npy"), H)
                np.save(os.path.join(tmpdir, f"{i}_pos.npy"), P)

            info = {
                "rho": np.ones(n, dtype=np.float32),
                "segSites": np.full(n, 10),
                "numReps": n,
                "hotWin": np.array([0, 3, 4, 5, 10, 100], dtype=np.float32),
            }
            pickle.dump(info, open(os.path.join(tmpdir, "info.p"), "wb"))

            gen = SequenceBatchGenerator(
                treesDirectory=tmpdir,
                targetNormalization='zscore',
                batchSize=16,
                maxLen=20,
                hotspots=True,
                seed=42,
            )
            result = gen.normalizeTargetsBinaryClass()
            expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
            np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Full batch generation (__getitem__)
# ---------------------------------------------------------------------------

class TestGetItem:
    def test_batch_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, n=100, nSamps=10, maxLen=50,
                                  frameWidth=5)
            x, y = gen.__getitem__(0)
            haps, pos = x
            # haps is 3D (batch, snps, samples), pos is 2D (batch, snps)
            assert haps.ndim == 3
            assert pos.ndim == 2
            assert y.ndim == 2
            assert haps.shape[0] == pos.shape[0] == y.shape[0]  # batch dim

    def test_all_batches_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = _make_generator(tmpdir, n=50, maxLen=50)
            for i in range(len(gen)):
                x, y = gen.__getitem__(i)
                assert x[0].shape[0] > 0
