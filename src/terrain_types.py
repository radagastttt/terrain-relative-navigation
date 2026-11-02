from __future__ import annotations

import numpy as np


def make_flat_dem(rows: int = 300, cols: int = 300,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dem = 100.0 + rng.normal(0, 0.3, (rows, cols))
    return dem.astype(np.float32)


def make_hilly_dem(rows: int = 300, cols: int = 300,
                   seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    X, Y = np.meshgrid(x, y)
    dem = (
        100.0
        + 40.0 * np.sin(X) * np.cos(Y)
        + 20.0 * np.sin(3 * X + 1.0) * np.cos(2 * Y)
    )
    dem += rng.normal(0, 2.0, dem.shape)
    return dem.astype(np.float32)


def make_mountainous_dem(rows: int = 300, cols: int = 300,
                         seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    X, Y = np.meshgrid(x, y)
    dem = (
        100.0
        + 80.0 * np.sin(X) * np.cos(Y)
        + 30.0 * np.sin(3 * X + 1.2) * np.cos(2 * Y)
        + 15.0 * np.sin(11 * X) * np.sin(9 * Y)
        + 8.0 * np.sin(23 * X + 0.5) * np.cos(17 * Y)
    )
    dem += rng.normal(0, 2.0, dem.shape)
    return dem.astype(np.float32)


DEM_GENERATORS = {
    "flat": make_flat_dem,
    "hilly": make_hilly_dem,
    "mountainous": make_mountainous_dem,
}


def get_dem(kind: str, rows: int = 300, cols: int = 300,
            seed: int = 42) -> np.ndarray:
    if kind not in DEM_GENERATORS:
        raise ValueError(f"Unknown terrain kind: {kind}")
    return DEM_GENERATORS[kind](rows, cols, seed)
