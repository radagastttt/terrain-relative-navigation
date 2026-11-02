from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.terrain_map import TerrainMap, make_synthetic_dem
from src.dem_loader import load_real_dem, terrain_params_from_transform
from src.trajectory import Trajectory
from src.measurement import Barometer
from src.particle_filter import TRNParticleFilter
from src.metrics import summarize
from src.visualization import plot_trajectory, plot_errors_comparison
from src.utils import load_config, set_seed, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_terrain(cfg: dict) -> TerrainMap:
    dcfg = cfg["dem"]
    if dcfg["source"] == "synthetic":
        dem = make_synthetic_dem(
            rows=dcfg["rows"],
            cols=dcfg["cols"],
            seed=cfg["seed"],
        )
        origin_xy = tuple(dcfg["origin_xy"])
        dx, dy = dcfg["dx"], dcfg["dy"]
    elif dcfg["source"] == "geotiff":
        path = Path(dcfg["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        dem, transform, crs = load_real_dem(str(path))
        origin_xy, dx, dy = terrain_params_from_transform(transform, dem.shape)
    else:
        raise NotImplementedError(
            f"DEM source '{dcfg['source']}' пока не поддержан"
        )
    return TerrainMap(dem, dx=dx, dy=dy, origin_xy=origin_xy)


def inject_outliers(z: np.ndarray, indices: list[int],
                    magnitude: float) -> np.ndarray:
    z = z.copy()
    for i in indices:
        if 0 <= i < len(z):
            z[i] += magnitude
    return z


def run_once(
    cfg: dict,
    logger,
    sigma_meas: float | None = None,
    sigma_proc: float | None = None,
    outlier_prob: float | None = None,
    n_particles: int | None = None,
    inject_at: list[int] | None = None,
    inject_magnitude: float = 50.0,
    seed_offset: int = 0,
    tag: str = "run",
    burn_in: int = 10,
    init_sigma_xy: float | None = None,
    init_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> dict:
    run_seed = cfg["seed"] + seed_offset

    terrain = build_terrain(cfg)

    tr = cfg["trajectory"]
    traj = Trajectory.from_straight(
        start_xy=tr["start_xy"],
        speed=tr["speed"],
        heading_deg=tr["heading_deg"],
        duration=tr["duration"],
        dt=tr["dt"],
    )
    if not all(terrain.in_bounds(x, y) for x, y in traj.positions_xy):
        raise ValueError("Траектория выходит за границы DEM.")

    mcfg = cfg["measurement"]
    sigma_meas_val = mcfg["sigma_baro"] if sigma_meas is None else sigma_meas
    bias_val = mcfg["bias"]
    sigma_filter = float(np.sqrt(sigma_meas_val ** 2 + bias_val ** 2))

    baro = Barometer(
        sigma=sigma_meas_val,
        bias=bias_val,
        outlier_prob=mcfg["outlier_prob"] if outlier_prob is None else outlier_prob,
        outlier_scale=mcfg["outlier_scale"],
        seed=run_seed,
    )
    true_heights = traj.true_heights(terrain)
    measured = baro.measure_many(true_heights)

    if inject_at:
        measured = inject_outliers(measured, inject_at, inject_magnitude)

    pcfg = cfg["particle_filter"]
    pf = TRNParticleFilter(
        terrain=terrain,
        n_particles=pcfg["n_particles"] if n_particles is None else n_particles,
        init_xy=np.asarray(tr["start_xy"], dtype=float) + np.asarray(init_offset_xy),
        init_sigma_xy=pcfg["init_sigma_xy"] if init_sigma_xy is None else init_sigma_xy,
        sigma_meas=sigma_filter,
        sigma_proc=pcfg["sigma_proc"] if sigma_proc is None else sigma_proc,
        resample_threshold=pcfg["resample_threshold"],
        seed=run_seed,
    )

    n = len(traj)
    history_true = np.empty((n, 2))
    history_est = np.empty((n, 2))
    errors = np.empty(n)
    neff_history = np.empty(n)
    n_gated = 0

    for i in range(n):
        v = traj.velocities_xy[i]
        dt_step = traj.times[i] - traj.times[i - 1] if i > 0 else tr["dt"]

        pf.predict(v, dt_step)
        pf.update(float(measured[i]))
        if pf.last_gated:
            n_gated += 1
        pf.maybe_resample()

        est = pf.estimate()
        history_true[i] = traj.positions_xy[i]
        history_est[i] = est
        errors[i] = np.linalg.norm(est - traj.positions_xy[i])
        neff_history[i] = pf.neff()

    summary = summarize(errors, burn_in=burn_in, convergence_threshold=3.0)

    logger.info(
        f"[{tag}] RMSE={summary['rmse']:.2f} m  "
        f"RMSE*={summary['rmse_after']:.2f} m  "
        f"max={summary['max']:.2f} m  "
        f"final={summary['final']:.2f} m  "
        f"conv_rel={summary['convergence_step_relative']}  "
        f"gated={n_gated}/{n}"
    )

    return {
        "terrain": terrain,
        "traj": traj,
        "true_heights": true_heights,
        "measured": measured,
        "history_true": history_true,
        "history_est": history_est,
        "errors": errors,
        "neff": neff_history,
        "rmse": summary["rmse"],
        "rmse_after": summary["rmse_after"],
        "max_err": summary["max"],
        "final_err": summary["final"],
        "convergence_step": summary["convergence_step"],
        "convergence_step_relative": summary["convergence_step_relative"],
        "convergence_step_stable": summary["convergence_step_stable"],
        "n_gated": n_gated,
        "tag": tag,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "figures"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = setup_logger("trn", str(PROJECT_ROOT / "results" / "logs" / "run.log"))

    out_dir = Path(args.out)
    n_steps = int(cfg["trajectory"]["duration"] / cfg["trajectory"]["dt"])

    logger.info("=== Стадия 0: идеальные условия (σ_meas=0.2) ===")
    res_ideal = run_once(
        cfg, logger,
        sigma_meas=0.2, sigma_proc=0.1,
        n_particles=5000, seed_offset=1,
        tag="ideal", burn_in=5,
    )
    plot_trajectory(res_ideal, out_dir / "stage0_ideal.png")

    logger.info("=== Стадия 1: σ_meas=0.5 м ===")
    res_noisy = run_once(
        cfg, logger,
        sigma_meas=0.5, sigma_proc=0.1,
        n_particles=5000, seed_offset=2,
        tag="noisy_0.5", burn_in=10,
    )
    plot_trajectory(res_noisy, out_dir / "stage1_noisy.png")

    logger.info("=== Стадия 2: σ_meas=1.0 м ===")
    res_proc = run_once(
        cfg, logger,
        sigma_meas=1.0, sigma_proc=0.1,
        n_particles=5000, seed_offset=3,
        tag="noisy_1.0", burn_in=10,
    )
    plot_trajectory(res_proc, out_dir / "stage2_process.png")

    logger.info("=== Стадия 3: σ_meas=3 м, 3 выброса ===")
    inject_at = [n_steps // 4, n_steps // 2, (3 * n_steps) // 4]
    res_real = run_once(
        cfg, logger,
        sigma_meas=3.0, sigma_proc=0.1,
        n_particles=5000, seed_offset=4,
        inject_at=inject_at, inject_magnitude=30.0,
        tag="outliers", burn_in=10,
    )
    plot_trajectory(res_real, out_dir / "stage3_outliers.png")

    logger.info("=== Стадия 4: чистая модель (bias=0, out=0) ===")
    cfg_clean = {
        **cfg,
        "measurement": {**cfg["measurement"], "bias": 0.0, "outlier_prob": 0.0},
    }
    res_clean = run_once(
        cfg_clean, logger,
        sigma_meas=0.5, sigma_proc=0.1,
        n_particles=5000, seed_offset=10,
        tag="clean", burn_in=10,
    )
    plot_trajectory(res_clean, out_dir / "stage4_clean.png")

    plot_errors_comparison(
        [res_ideal, res_noisy, res_proc, res_real, res_clean],
        out_dir / "errors_comparison.png",
    )

    print("\n" + "=" * 100)
    print(f"{'Стадия':<12}{'RMSE, м':>10}{'RMSE*':>10}"
          f"{'max, м':>10}{'final, м':>10}{'conv_rel':>10}{'gated':>10}")
    print("-" * 100)
    for r in (res_ideal, res_noisy, res_proc, res_real, res_clean):
        print(f"{r['tag']:<12}"
              f"{r['rmse']:>10.2f}"
              f"{r['rmse_after']:>10.2f}"
              f"{r['max_err']:>10.2f}"
              f"{r['final_err']:>10.2f}"
              f"{str(r['convergence_step_relative']):>10}"
              f"{str(r['n_gated']):>10}")
    print("=" * 100)


if __name__ == "__main__":
    main()
