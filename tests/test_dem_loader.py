import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from src.dem_loader import load_real_dem, terrain_params_from_transform
from src.terrain_map import TerrainMap


@pytest.fixture
def utm_tif(tmp_path):
    data = (10 * np.arange(3)[:, None] + np.arange(4)[None, :]).astype(np.float32)
    data[0, 3] = -9999.0
    path = tmp_path / "dem.tif"
    transform = from_origin(500000.0, 4700090.0, 30.0, 30.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=3, width=4, count=1,
        dtype="float32", crs="EPSG:32635", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def test_load_flips_and_fills_nodata(utm_tif):
    dem, transform, crs = load_real_dem(str(utm_tif))
    assert crs.to_epsg() == 32635
    assert dem.shape == (3, 4)
    assert dem[0, 0] == pytest.approx(20.0)
    assert dem[2, 0] == pytest.approx(0.0)
    assert np.isfinite(dem).all()


def test_terrain_params_and_heights(utm_tif):
    dem, transform, _ = load_real_dem(str(utm_tif))
    origin, dx, dy = terrain_params_from_transform(transform, dem.shape)
    assert (dx, dy) == (30.0, 30.0)
    assert origin == pytest.approx((500015.0, 4700015.0))
    tm = TerrainMap(dem, dx=dx, dy=dy, origin_xy=origin)
    assert tm.height_at(500075.0, 4700075.0) == pytest.approx(2.0)
    assert tm.height_at(500015.0, 4700015.0) == pytest.approx(20.0)


def test_geographic_crs_rejected(tmp_path):
    path = tmp_path / "geo.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(24.0, 43.0, 0.001, 0.001),
    ) as dst:
        dst.write(np.zeros((2, 2), np.float32), 1)
    with pytest.raises(ValueError):
        load_real_dem(str(path))
