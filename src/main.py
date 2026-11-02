import numpy as np
import scipy
import matplotlib
from scipy.ndimage import map_coordinates
from terrain_map import TerrainMap, make_synthetic_dem
from zoom import disable_rect_zoom, enable_scroll_zoom, limit_view_to_image

small = np.array([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]], dtype=np.float32)
tm = TerrainMap(small, dx=1.0, dy=1.0)

print(tm.height_at(0, 0))
print(tm.height_at(1, 1))
print(tm.height_at(1.5, 1.5))
print(tm.height_at(2, 2))

print(tm.height_at(5, 5))

xs = np.array([0.0, 1.0, 1.5, 2.0])
ys = np.array([0.0, 1.0, 1.5, 2.0])
print(tm.height_at_many(xs, ys))

import argparse
import sys
from pathlib import Path

from utils import load_config
from dem_loader import load_real_dem, terrain_params_from_transform
from dem_dialog import ask_dem_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
parser.add_argument("--dem", help="путь к GeoTIFF (вместо dem.path из config.yaml)")
parser.add_argument("--synthetic", action="store_true", help="синтетический DEM")
parser.add_argument("--no-dialog", action="store_true",
                    help="не открывать диалог, взять DEM из config.yaml")
args = parser.parse_args()

dcfg = load_config(args.config)["dem"]
disable_rect_zoom()
chosen = None
if not (args.dem or args.synthetic or args.no_dialog):
    chosen = ask_dem_path(PROJECT_ROOT / "data" / "raw")
if args.dem:
    dem_path = Path(args.dem)
    if not dem_path.exists():
        dem_path = PROJECT_ROOT / args.dem
elif chosen is not None:
    dem_path = chosen
elif dcfg["source"] == "geotiff":
    dem_path = PROJECT_ROOT / dcfg["path"]
else:
    dem_path = None

if args.synthetic or dem_path is None:
    dem = make_synthetic_dem(rows=200, cols=200)
    tm_big = TerrainMap(dem, dx=30.0, dy=30.0, origin_xy=(0.0, 0.0))
    title = "Синтетический DEM"
else:
    try:
        dem, transform, crs = load_real_dem(str(dem_path))
    except Exception as e:
        sys.exit(f"Не удалось открыть DEM {dem_path}: {e}")
    origin_xy, dx, dy = terrain_params_from_transform(transform, dem.shape)
    tm_big = TerrainMap(dem, dx=dx, dy=dy, origin_xy=origin_xy)
    title = f"{dem_path.name} ({crs})"
print(title)
print("Размер карты, км:",
      round((tm_big.x_max - tm_big.x_min) / 1e3, 2), "x",
      round((tm_big.y_max - tm_big.y_min) / 1e3, 2))

n = 500
xs = np.linspace(tm_big.x_min, tm_big.x_max, n)
ys = np.linspace(tm_big.y_min, tm_big.y_max, n)
heights = tm_big.height_at_many(xs, ys)
dist_km = np.hypot(xs - xs[0], ys - ys[0]) / 1e3

extent = [(tm_big.x_min - tm_big.dx / 2) / 1e3, (tm_big.x_max + tm_big.dx / 2) / 1e3,
          (tm_big.y_min - tm_big.dy / 2) / 1e3, (tm_big.y_max + tm_big.dy / 2) / 1e3]

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(13, 5))
enable_scroll_zoom(fig)
plt.subplot(1, 2, 1)
plt.imshow(tm_big.dem, cmap='terrain', origin='lower', extent=extent)
plt.colorbar(label="Высота, м", shrink=0.85)
plt.plot(xs / 1e3, ys / 1e3, 'r-', lw=1, label='Профиль')
plt.xlabel("X (восток), км")
plt.ylabel("Y (север), км")
plt.title(f"DEM + линия профиля\n{title}", fontsize=10)
plt.legend()
limit_view_to_image(plt.gca())

plt.subplot(1, 2, 2)
plt.plot(dist_km, heights)
plt.xlabel("Расстояние вдоль линии, км")
plt.ylabel("Высота, м")
plt.title("Профиль высот вдоль линии")
plt.grid(True)
plt.tight_layout()
plt.show()
