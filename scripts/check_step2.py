import numpy as np
import matplotlib.pyplot as plt

from src.terrain_map import TerrainMap, make_synthetic_dem
from src.trajectory import Trajectory
from src.measurement import Barometer
from src.utils import load_config


def main():
    cfg = load_config("config.yaml")

    dem = make_synthetic_dem(
        rows=cfg["dem"]["rows"], cols=cfg["dem"]["cols"], seed=cfg["seed"]
    )
    terrain = TerrainMap(
        dem,
        dx=cfg["dem"]["dx"], dy=cfg["dem"]["dy"],
        origin_xy=tuple(cfg["dem"]["origin_xy"]),
    )

    tr = cfg["trajectory"]
    traj = Trajectory.from_straight(
        start_xy=tr["start_xy"], speed=tr["speed"],
        heading_deg=tr["heading_deg"], duration=tr["duration"], dt=tr["dt"],
    )

    true_h = traj.true_heights(terrain)
    baro = Barometer(
        sigma=cfg["measurement"]["sigma_baro"],
        bias=cfg["measurement"]["bias"],
        outlier_prob=cfg["measurement"]["outlier_prob"],
        outlier_scale=cfg["measurement"]["outlier_scale"],
        seed=cfg["seed"],
    )
    z = baro.measure_many(true_h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(
        dem, cmap="terrain", origin="lower",
        extent=[terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max],
    )
    axes[0].plot(traj.positions_xy[:, 0], traj.positions_xy[:, 1], "r-", lw=2)
    axes[0].plot(traj.positions_xy[0, 0], traj.positions_xy[0, 1], "go", ms=8, label="start")
    axes[0].plot(traj.positions_xy[-1, 0], traj.positions_xy[-1, 1], "bs", ms=8, label="end")
    axes[0].set_title("DEM + траектория")
    axes[0].legend()

    axes[1].plot(traj.times, true_h, "k-", lw=1.5, label="Истинная высота")
    axes[1].plot(traj.times, z, "r.", ms=3, alpha=0.7, label="Измерения")
    axes[1].set_xlabel("t, с")
    axes[1].set_ylabel("Высота, м")
    axes[1].set_title("Профиль высот")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/step2_check.png", dpi=150)
    print("RMSE измерений:", float(np.sqrt(((z - true_h) ** 2).mean())))


if __name__ == "__main__":
    main()
