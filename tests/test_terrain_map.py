import numpy as np
import pytest

from src.terrain_map import TerrainMap, make_synthetic_dem


@pytest.fixture
def small_tm():
    dem = np.array([[10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90]], dtype=np.float32)
    return TerrainMap(dem, dx=1.0, dy=1.0)


def test_height_at_nodes(small_tm):
    assert small_tm.height_at(0, 0) == pytest.approx(10.0)
    assert small_tm.height_at(1, 1) == pytest.approx(50.0)
    assert small_tm.height_at(2, 2) == pytest.approx(90.0)


def test_bilinear_center(small_tm):
    assert small_tm.height_at(1.5, 1.5) == pytest.approx(70.0)


def test_out_of_bounds_returns_nan(small_tm):
    assert np.isnan(small_tm.height_at(5.0, 5.0))


def test_world_pixel_roundtrip(small_tm):
    x, y = 1.3, 0.7
    row, col = small_tm.world_to_pixel(x, y)
    x2, y2 = small_tm.pixel_to_world(row, col)
    assert x2 == pytest.approx(x)
    assert y2 == pytest.approx(y)


def test_height_at_many(small_tm):
    xs = np.array([0.0, 1.0, 1.5, 2.0])
    ys = np.array([0.0, 1.0, 1.5, 2.0])
    h = small_tm.height_at_many(xs, ys)
    np.testing.assert_allclose(h, [10, 50, 70, 90], atol=1e-5)


def test_synthetic_dem_shape():
    dem = make_synthetic_dem(rows=100, cols=150)
    assert dem.shape == (100, 150)
    assert dem.dtype == np.float32
