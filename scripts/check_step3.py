from pathlib import Path

import numpy as np

from src.terrain_map import TerrainMap, make_synthetic_dem
from src.trajectory import Trajectory
from src.measurement import Barometer
from src.particle_filter import TRNParticleFilter
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_profile_uniqueness(cfg: dict, terrain: TerrainMap, traj: Trajectory):
    print("\n" + "=" * 70)
    print("Гипотеза 1: информативность рельефа под траекторией")
    print("=" * 70)

    h = traj.true_heights(terrain)
    print(f"Профиль высот:")
    print(f"  min  : {h.min():.2f} м")
    print(f"  max  : {h.max():.2f} м")
    print(f"  mean : {h.mean():.2f} м")
    print(f"  std  : {h.std():.2f} м")

    head = np.deg2rad(cfg["trajectory"]["heading_deg"])
    perp = np.array([-np.sin(head), np.cos(head)])

    print("\nУникальность профиля при перпендикулярном сдвиге:")
    print(f"  {'shift, м':>10} | {'RMSE(profile diff), м':>22}")
    print("  " + "-" * 36)

    for shift in [5, 10, 20, 30, 50, 100, 200]:
        pos_shift = traj.positions_xy + shift * perp
        h_shift = terrain.height_at_many(pos_shift[:, 0], pos_shift[:, 1])
        valid = np.isfinite(h_shift) & np.isfinite(h)
        if valid.sum() == 0:
            print(f"  {shift:>10} | {'(вне карты)':>22}")
            continue
        rmse = float(np.sqrt(((h[valid] - h_shift[valid]) ** 2).mean()))
        print(f"  {shift:>10} | {rmse:>22.2f}")

    print("\nИнтерпретация:")
    print("  Если RMSE при shift=30 м сопоставим с sigma_baro (~3 м) —")
    print("  рельеф неинформативен, фильтр не сможет локализоваться точно.")


def check_param_window(cfg: dict, terrain: TerrainMap, traj: Trajectory):
    print("\n" + "=" * 70)
    print("Гипотеза 2: сетка sigma_proc × sigma_meas (с реальными шумными измерениями)")
    print("=" * 70)

    true_h = traj.true_heights(terrain)

    sigma_proc_grid = [0.05, 0.1, 0.5, 1.0, 2.0]
    sigma_meas_grid = [0.05, 0.2, 0.5, 1.0, 3.0]

    header = f"{'σ_proc':>8} | " + " | ".join(
        f"{f'σ_meas={s}':>12}" for s in sigma_meas_grid
    )
    print(header)
    print("-" * len(header))

    for sp in sigma_proc_grid:
        row = f"{sp:>8.2f} | "
        cells = []
        for sm in sigma_meas_grid:
            baro = Barometer(
                sigma=sm,
                bias=0.0,
                outlier_prob=0.0,
                bias_sigma=0.0,
                seed=cfg["seed"],
            )
            measured = baro.measure_many(true_h)

            n_part = cfg.get("particle_filter", {}).get("n_particles", 2000)
            pf = TRNParticleFilter(
                terrain=terrain,
                n_particles=n_part,
                init_xy=cfg["trajectory"]["start_xy"],
                init_sigma_xy=cfg.get("particle_filter", {}).get("init_sigma_xy", 50.0),
                sigma_meas=sm,
                sigma_proc=sp,
                seed=cfg["seed"],
            )
            errs = []
            for i in range(len(traj)):
                pf.predict(traj.velocities_xy[i], cfg["trajectory"]["dt"])
                pf.update(float(measured[i]))
                pf.maybe_resample()
                errs.append(
                    np.linalg.norm(pf.estimate() - traj.positions_xy[i])
                )
            errs = np.array(errs)
            rmse_after = float(np.sqrt((errs[10:] ** 2).mean()))
            cells.append(f"{rmse_after:>12.2f}")
        row += " | ".join(cells)
        print(row)

    print("\nЗначения — RMSE* (м), посчитанное после burn-in 10 шагов.")
    print("Измерения содержат шум с σ = σ_meas (bias=0, без выбросов).")

def check_neff_dynamics(cfg: dict, terrain: TerrainMap, traj: Trajectory):
    print("\n" + "=" * 70)
    print("Гипотеза 3: динамика Neff на идеальных измерениях")
    print("=" * 70)

    true_h = traj.true_heights(terrain)

    for sigma_meas, sigma_proc in [(0.01, 0.01), (0.5, 0.5), (3.0, 0.5)]:
        n_part = cfg.get("particle_filter", {}).get("n_particles", 2000)
        pf = TRNParticleFilter(
            terrain=terrain,
            n_particles=n_part,
            init_xy=cfg["trajectory"]["start_xy"],
            init_sigma_xy=cfg.get("particle_filter", {}).get("init_sigma_xy", 50.0),
            sigma_meas=sigma_meas,
            sigma_proc=sigma_proc,
            seed=cfg["seed"],
        )
        neffs = []
        errs = []
        for i in range(len(traj)):
            pf.predict(traj.velocities_xy[i], cfg["trajectory"]["dt"])
            pf.update(float(true_h[i]))
            pf.maybe_resample()
            neffs.append(pf.neff())
            errs.append(np.linalg.norm(pf.estimate() - traj.positions_xy[i]))
        neffs = np.array(neffs)
        errs = np.array(errs)

        print(f"\nσ_meas={sigma_meas}, σ_proc={sigma_proc}")
        print(f"  Neff first 5 : {neffs[:5].round(1).tolist()}")
        print(f"  Neff last 5  : {neffs[-5:].round(1).tolist()}")
        print(f"  Neff min     : {neffs.min():.1f}  (из {pf.n})")
        print(f"  RMSE* (after 10) = {np.sqrt((errs[10:] ** 2).mean()):.2f} м")
        print(f"  final error      = {errs[-1]:.2f} м")


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))

    dem = make_synthetic_dem(
        cfg["dem"]["rows"], cfg["dem"]["cols"], seed=cfg["seed"]
    )
    terrain = TerrainMap(
        dem,
        dx=cfg["dem"]["dx"],
        dy=cfg["dem"]["dy"],
        origin_xy=tuple(cfg["dem"]["origin_xy"]),
    )

    tr = cfg["trajectory"]
    traj = Trajectory.from_straight(
        start_xy=tr["start_xy"],
        speed=tr["speed"],
        heading_deg=tr["heading_deg"],
        duration=tr["duration"],
        dt=tr["dt"],
    )

    check_profile_uniqueness(cfg, terrain, traj)
    check_param_window(cfg, terrain, traj)
    check_neff_dynamics(cfg, terrain, traj)


if __name__ == "__main__":
    main()
