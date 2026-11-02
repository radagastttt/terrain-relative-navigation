from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def make_synthetic_dem(rows: int = 200, cols: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    X, Y = np.meshgrid(x, y)

    dem = (
        100.0
        + 80.0 * np.sin(X) * np.cos(Y)
        + 30.0 * np.sin(3 * X + 1.2) * np.cos(2 * Y)
        + 15.0 * np.sin(11 * X) * np.sin(9 * Y)
        + 8.0  * np.sin(23 * X + 0.5) * np.cos(17 * Y)
    )
    dem += rng.normal(0, 2.0, dem.shape)
    return dem.astype(np.float32)


def load_dem_geotiff(path: str):
    import rasterio
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    return dem, transform, crs


def load_dem_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


class TerrainMap:
    def __init__(self, dem: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                 origin_xy: tuple[float, float] = (0.0, 0.0)):
        self.dem = dem.astype(np.float32)
        self.dx = float(dx)
        self.dy = float(dy)
        self.origin_x, self.origin_y = origin_xy
        self.rows, self.cols = dem.shape

        self.x_min = self.origin_x
        self.x_max = self.origin_x + (self.cols - 1) * self.dx
        self.y_min = self.origin_y
        self.y_max = self.origin_y + (self.rows - 1) * self.dy

    def world_to_pixel(self, x, y):
        col = (x - self.origin_x) / self.dx
        row = (y - self.origin_y) / self.dy
        return row, col

    def pixel_to_world(self, row, col):
        x = self.origin_x + col * self.dx
        y = self.origin_y + row * self.dy
        return x, y

    def in_bounds(self, x, y, margin: float = 0.0) -> bool:
        return (self.x_min - margin <= x <= self.x_max + margin and
                self.y_min - margin <= y <= self.y_max + margin)

    def height_at(self, x, y, order: int = 1) -> float:
        if not self.in_bounds(x, y, margin=1e-6):
            return np.nan
        row, col = self.world_to_pixel(x, y)
        return float(map_coordinates(self.dem, [[row], [col]],
                                     order=order, mode='nearest')[0])

    def height_at_many(self, xs, ys, order: int = 1) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        cols = (xs - self.origin_x) / self.dx
        rows = (ys - self.origin_y) / self.dy
        coords = np.vstack([rows, cols])
        heights = map_coordinates(self.dem, coords, order=order, mode='nearest')
        out = (
            (xs < self.x_min - 1e-6) | (xs > self.x_max + 1e-6) |
            (ys < self.y_min - 1e-6) | (ys > self.y_max + 1e-6)
        )
        heights = heights.astype(np.float32)
        heights[out] = np.nan
        return heights
