from __future__ import annotations

import numpy as np


def load_real_dem(path: str, fill_nodata: bool = True):
    import rasterio

    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if crs is not None and not crs.is_projected:
        raise ValueError(
            f"DEM {path} в географической СК ({crs}); нужна метрическая (UTM)."
        )
    if transform.b != 0 or transform.d != 0:
        raise ValueError("Повёрнутые растры не поддерживаются.")

    if nodata is not None:
        dem[dem == nodata] = np.nan
    if fill_nodata and np.isnan(dem).any():
        dem = _fill_nan_nearest(dem)

    if transform.e < 0:
        dem = np.flipud(dem).copy()
    return dem, transform, crs


def terrain_params_from_transform(transform, shape):
    rows, _ = shape
    dx = abs(transform.a)
    dy = abs(transform.e)
    origin_x = transform.c + transform.a / 2.0
    if transform.e < 0:
        origin_y = transform.f + transform.e * (rows - 0.5)
    else:
        origin_y = transform.f + transform.e / 2.0
    return (float(origin_x), float(origin_y)), float(dx), float(dy)


def _fill_nan_nearest(dem: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(dem)
    if mask.all():
        raise ValueError("DEM не содержит валидных значений.")
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return dem[tuple(idx)]
