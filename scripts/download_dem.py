from __future__ import annotations

import argparse
import math
import os
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BBOX = (24.75, 42.60, 25.05, 42.80)
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "central_balkan_cop30_utm35n.tif"
DST_CRS = "EPSG:32635"
RESOLUTION = 30.0

OPENTOPO_URL = (
    "https://portal.opentopography.org/API/globaldem?demtype=COP30"
    "&west={w}&south={s}&east={e}&north={n}&outputFormat=GTiff&API_Key={key}"
)
AWS_TILE = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"
)


def _aws_tile_urls(bbox):
    w, s, e, n = bbox
    urls = []
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            lat_s = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lon_s = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            urls.append(AWS_TILE.format(lat=lat_s, lon=lon_s))
    return urls


def fetch_geographic(bbox, api_key: str | None, tmp_dir: Path) -> Path:
    import rasterio
    from rasterio.merge import merge

    out = tmp_dir / "dem_wgs84.tif"
    if api_key:
        w, s, e, n = bbox
        url = OPENTOPO_URL.format(w=w, s=s, e=e, n=n, key=api_key)
        print("OpenTopography COP30 ...")
        urllib.request.urlretrieve(url, out)
        return out

    print("Copernicus GLO-30 (AWS Open Data) ...")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        sources = [rasterio.open(f"/vsicurl/{u}") for u in _aws_tile_urls(bbox)]
        try:
            mosaic, transform = merge(sources, bounds=bbox)
            profile = sources[0].profile
        finally:
            for src in sources:
                src.close()
    profile.update(
        driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
        transform=transform, count=1, compress="deflate",
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mosaic[:1])
    return out


def reproject_to_utm(src_path: Path, dst_path: Path,
                     dst_crs: str = DST_CRS, resolution: float = RESOLUTION):
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution,
        )
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile.copy()
        profile.update(
            driver="GTiff", crs=dst_crs, transform=transform,
            width=width, height=height, dtype="float32",
            nodata=nodata, compress="deflate",
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=dst_crs,
                src_nodata=src.nodata, dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--api-key", default=os.environ.get("OPENTOPOGRAPHY_API_KEY"))
    args = parser.parse_args()

    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        geo = fetch_geographic(tuple(args.bbox), args.api_key, Path(tmp))
        reproject_to_utm(geo, out)

    import rasterio
    with rasterio.open(out) as ds:
        dem = ds.read(1, masked=True)
        print(f"Сохранено: {out}")
        print(f"CRS={ds.crs}  size={ds.width}x{ds.height}  res={ds.res}")
        print(f"bounds={tuple(round(v, 1) for v in ds.bounds)}")
        print(f"высоты: {dem.min():.0f}..{dem.max():.0f} м")


if __name__ == "__main__":
    main()
