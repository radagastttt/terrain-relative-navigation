from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

from src.terrain_map import TerrainMap, make_synthetic_dem
from src.trajectory import Trajectory
from src.measurement import Barometer
from src.particle_filter import TRNParticleFilter
from src.metrics import summarize
from src.utils import load_config, set_seed, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_once(cfg, n_particles, logger, clean: bool = False):
    dem = make_synthetic_dem(cfg["dem"]["rows"], cfg["dem"]["cols"],
                             seed=cfg["seed"])
    terrain = TerrainMap(dem, cfg["dem"]["dx"], cfg["dem"]["dy"])

    tr = cfg["trajectory"]
    traj = Trajectory.from_straight(
        tr["start_xy"], tr["speed"], tr["heading_deg"],
        tr["duration"], tr["dt"],
    )

    mcfg = cfg["measurement"]
    if clean:
        bias = 0.0
        outlier_prob = 0.0
        sigma_filter = mcfg["sigma_baro"]
    else:
        bias = mcfg["bias"]
        outlier_prob = mcfg["outlier_prob"]
        sigma_filter = float(np.sqrt(mcfg["sigma_baro"] ** 2 + bias ** 2))

    baro = Barometer(sigma=mcfg["sigma_baro"], bias=bias,
                     outlier_prob=outlier_prob,
                     outlier_scale=mcfg["outlier_scale"], seed=cfg["seed"])
    measured = baro.measure_many(traj.true_heights(terrain))

    pcfg = cfg["particle_filter"]
    pf = TRNParticleFilter(
        terrain, n_particles=n_particles,
        init_xy=tr["start_xy"], init_sigma_xy=pcfg["init_sigma_xy"],
        sigma_meas=sigma_filter, sigma_proc=pcfg["sigma_proc"],
        seed=cfg["seed"],
    )

    t0 = time.perf_counter()
    errs = []
    for i in range(len(traj)):
        pf.predict(traj.velocities_xy[i], tr["dt"])
        pf.update(float(measured[i]))
        pf.maybe_resample()
        errs.append(np.linalg.norm(pf.estimate() - traj.positions_xy[i]))
    elapsed = time.perf_counter() - t0

    summary = summarize(np.array(errs), burn_in=15, convergence_threshold=3.0)
    tag = "clean" if clean else "real"
    logger.info(
        f"[{tag} n={n_particles:>6}] RMSE*={summary['rmse_after']:.2f} m  "
        f"final={summary['final']:.2f} m  time={elapsed:.3f} s"
    )
    return {
        "n": n_particles,
        "rmse_after": summary["rmse_after"],
        "final": summary["final"],
        "time": elapsed,
        "convergence_step": summary["convergence_step"],
        "convergence_step_relative": summary["convergence_step_relative"],
        "convergence_step_stable": summary["convergence_step_stable"],
    }


def run_sweep(cfg, ns, logger, clean: bool):
    results = [run_once(cfg, n, logger, clean=clean) for n in ns]
    return results


def plot_sweep(results_clean, results_real, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([r["n"] for r in results_clean],
            [r["rmse_after"] for r in results_clean],
            "o-", color="b", label="clean (bias=0, no outliers)")
    ax.plot([r["n"] for r in results_real],
            [r["rmse_after"] for r in results_real],
            "s--", color="r", label="realistic (bias=0.5, outliers 2%)")
    ax.set_xscale("log")
    ax.set_xlabel("n_particles (log)")
    ax.set_ylabel("RMSE*, м")
    ax.set_title("Влияние n_particles на RMSE*")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time(results, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([r["n"] for r in results],
            [r["time"] for r in results],
            "o-", color="g")
    ax.set_xscale("log")
    ax.set_xlabel("n_particles (log)")
    ax.set_ylabel("Время прогона, с")
    ax.set_title("Время работы фильтра vs n_particles")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def print_table(results_clean, results_real):
    print("\n" + "=" * 90)
    print(f"{'n':>8}{'RMSE* (clean)':>16}{'RMSE* (real)':>16}"
          f"{'final (real)':>16}{'time, с':>14}")
    print("-" * 90)
    for rc, rr in zip(results_clean, results_real):
        print(f"{rc['n']:>8}"
              f"{rc['rmse_after']:>16.2f}"
              f"{rr['rmse_after']:>16.2f}"
              f"{rr['final']:>16.2f}"
              f"{rr['time']:>14.3f}")
    print("=" * 90)


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    set_seed(cfg["seed"])
    logger = setup_logger("trn_np",
                          str(PROJECT_ROOT / "results" / "logs" / "n_particles.log"))

    out_dir = PROJECT_ROOT / "results" / "figures" / "n_particles"
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = [200, 500, 1000, 2000, 5000, 10000, 20000]

    logger.info("=== Чистый сценарий (bias=0, outliers=0) ===")
    results_clean = run_sweep(cfg, ns, logger, clean=True)

    logger.info("=== Реалистичный сценарий (bias=0.5, outliers 2%) ===")
    results_real = run_sweep(cfg, ns, logger, clean=False)

    plot_sweep(results_clean, results_real,
               out_dir / "n_particles_vs_rmse.png")
    plot_time(results_real, out_dir / "n_particles_vs_time.png")
    print_table(results_clean, results_real)


if __name__ == "__main__":
    main()
