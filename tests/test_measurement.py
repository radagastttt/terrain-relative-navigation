import numpy as np
import pytest

from src.measurement import Barometer


def test_noiseless_equals_truth():
    b = Barometer(sigma=0.0, bias=0.0, outlier_prob=0.0,
                  bias_sigma=0.0, seed=0)
    true_h = np.array([100.0, 101.0, 102.0, 103.0])
    z = b.measure_many(true_h)
    np.testing.assert_allclose(z, true_h, atol=1e-9)


def test_noise_std_matches_sigma():
    b = Barometer(sigma=3.0, bias=0.0, outlier_prob=0.0,
                  bias_sigma=0.0, seed=42)
    true_h = np.full(10_000, 500.0, dtype=np.float32)
    z = b.measure_many(true_h)
    err = z - true_h
    assert abs(err.std() - 3.0) < 0.3


def test_bias_shift():
    b = Barometer(sigma=0.0, bias=10.0, outlier_prob=0.0,
                  bias_sigma=0.0, bias_ar1=1.0, seed=0)
    true_h = np.array([100.0, 200.0, 300.0])
    z = b.measure_many(true_h)
    np.testing.assert_allclose(z - true_h, 10.0, atol=1e-6)


def test_outliers_present():
    b = Barometer(sigma=1.0, bias=0.0, outlier_prob=0.1,
                  outlier_scale=20.0, bias_sigma=0.0, seed=1)
    true_h = np.full(2000, 100.0, dtype=np.float32)
    z = b.measure_many(true_h)
    err = np.abs(z - true_h)
    assert (err > 10.0).sum() > 0
