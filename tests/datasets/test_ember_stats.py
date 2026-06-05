"""
Regression test for EMBER feature-statistic computation.

Guards against the scale bug found during 500k validation: the old
sampled-float32 stats path overflowed an ultra-sparse, large-valued
feature's std to inf (silently zeroing the column) once the training
subset exceeded ~100k rows. The fixed path (_compute_feature_stats)
accumulates in float64 over the full subset, streamed in chunks.
"""

import numpy as np

from examples.datasets.ember import _compute_feature_stats


def _make_data(n=200_000, ndim=4, big=1e20, n_big=50, seed=0):
    """Synthetic features with one ultra-sparse, huge-valued column whose
    square overflows float32."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, ndim)).astype(np.float32)
    X[:, 0] = 0.0
    X[rng.choice(n, size=n_big, replace=False), 0] = big
    return X


def _old_sampled_float32_std(X, indices, n_stats=100_000, seed=0):
    """Reproduction of the old behavior: sample up to n_stats rows and
    compute std in the array's native float32 dtype."""
    rng = np.random.default_rng(seed)
    idx = indices
    if len(idx) > n_stats:
        idx = rng.choice(idx, size=n_stats, replace=False)
    sample = X[np.sort(idx)]
    # The float32 overflow is the behavior under test, so silence its warning.
    with np.errstate(over="ignore", invalid="ignore"):
        return sample.std(axis=0).astype(np.float32)


def test_old_path_overflows_sparse_feature():
    """The old path produces a non-finite std on the sparse feature."""
    X = _make_data()
    idx = np.arange(len(X))
    old_std = _old_sampled_float32_std(X, idx)
    assert not np.isfinite(old_std[0])


def test_fixed_path_is_finite_and_standardizes():
    """The fixed path is finite everywhere and properly standardizes the
    feature the old path zeroed."""
    X = _make_data()
    idx = np.arange(len(X))
    mean, std = _compute_feature_stats(X, idx)

    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    # The sparse feature gets a real, positive std (not zeroed, not inf).
    assert std[0] > 0.0

    col0 = (X[:, 0].astype(np.float64) - mean[0]) / std[0]
    assert np.isfinite(col0).all()
    # Properly standardized: unit-ish std over the full data.
    assert abs(col0.std() - 1.0) < 1e-3


def test_streaming_matches_numpy_reference():
    """float64 chunked streaming matches a direct float64 np.mean/np.std,
    and is invariant to chunk size."""
    X = _make_data(n=50_000, ndim=6)
    idx = np.arange(len(X))
    mean, std = _compute_feature_stats(X, idx, chunk=7_000)

    ref_mean = X.astype(np.float64).mean(axis=0)
    ref_std = X.astype(np.float64).std(axis=0)
    ref_std = np.where(ref_std < 1e-6, 1.0, ref_std)

    assert np.allclose(mean, ref_mean.astype(np.float32), rtol=1e-4, atol=1e-4)
    assert np.allclose(std, ref_std.astype(np.float32), rtol=1e-3, atol=1e-3)


def test_constant_feature_std_floor():
    """A constant feature gets std=1 (the eps floor), not 0."""
    X = np.zeros((1000, 3), dtype=np.float32)
    X[:, 1] = 5.0  # constant nonzero
    mean, std = _compute_feature_stats(X, np.arange(len(X)))
    assert np.allclose(std, 1.0)
    assert np.isclose(mean[1], 5.0)
