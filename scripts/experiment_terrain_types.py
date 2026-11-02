from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.terrain_map import TerrainMap
from src.trajectory import Trajectory
from src.measurement import Barometer
from src.particle_filter import TRNParticleFilter
from src.terrain_types import get_dem
from src.metrics import summarize
from src.visualization import plot_trajectory
from src.utils import load_config, set_seed, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_on_terrain(cfg, terrain_kind: str, logger):
    dem = get_dem(
        terrain_kind,
        rows=cfg["dem"]["rows"],
        cols=cfg["dem"]["cols"],
        seed=cfg["seed"],
    )
    terrain = TerrainMap(dem, cfg["dem"]["dx"], cfg["dem"]["dy"],
                         tuple(cfg["dem"]["origin_xy"]))

    tr = cfg["trajectory"]
    traj = Trajectory.from_straight(
        tr["start_xy"], tr["speed"], tr["heading_deg"],
        tr["duration"], tr["dt"],
    )

    mcfg = cfg["measurement"]
    pcfg = cfg["particle_filter"]
    sigma_meas = mcfg["sigma_baro"]
    bias = mcfg["bias"]
    sigma_filter = float(np.sqrt(sigma_meas ** 2 + bias ** 2))

    baro = Barometer(sigma=sigma_meas, bias=bias,
                     outlier_prob=mcfg["outlier_prob"],
                     outlier_scale=mcfg["outlier_scale"],
                     seed=cfg["seed"])
    true_h = traj.true_heights(terrain)
    measured = baro.measure_many(true_h)

    pf = TRNParticleFilter(
        terrain, n_particles=pcfg["n_particles"],
        init_xy=tr["start_xy"], init_sigma_xy=pcfg["init_sigma_xy"],
        sigma_meas=sigma_filter, sigma_proc=pcfg["sigma_proc"],
        seed=cfg["seed"],
    )

    errs, neffs = [], []
    hist_true, hist_est = [], []
    for i in range(len(traj)):
        pf.predict(traj.velocities_xy[i], tr["dt"])
        pf.update(float(measured[i]))
        pf.maybe_resample()
        est = pf.estimate()
        errs.append(np.linalg.norm(est - traj.positions_xy[i]))
        neffs.append(pf.neff())
        hist_true.append(traj.positions_xy[i])
        hist_est.append(est)

    errs = np.array(errs)
    summary = summarize(errs, burn_in=15, convergence_threshold=3.0)

    logger.info(
        f"[{terrain_kind}] RMSE*={summary['rmse_after']:.2f} m  "
        f"final={summary['final']:.2f} m  "
        f"conv_abs={summary['convergence_step']}  "
        f"conv_rel={summary['convergence_step_relative']}  "
        f"conv_stable={summary['convergence_step_stable']}"
    )

    return {
        "terrain_kind": terrain_kind,
        "terrain": terrain,
        "traj": traj,
        "history_true": np.array(hist_true),
        "history_est": np.array(hist_est),
        "errors": errs,
        "neff": np.array(neffs),
        "rmse": summary["rmse"],
        "rmse_after": summary["rmse_after"],
        "max_err": summary["max"],
        "final_err": summary["final"],
        "convergence_step": summary["convergence_step"],
        "convergence_step_relative": summary["convergence_step_relative"],
        "convergence_step_stable": summary["convergence_step_stable"],
        "tag": terrain_kind,
    }


def main():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    set_seed(cfg["seed"])
    logger = setup_logger("trn_terrain",
                          str(PROJECT_ROOT / "results" / "logs" / "terrain.log"))

    out_dir = PROJECT_ROOT / "results" / "figures" / "terrain_types"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for kind in ["flat", "hilly", "mountainous"]:
        res = run_on_terrain(cfg, kind, logger)
        results.append(res)
        plot_trajectory(res, out_dir / f"{kind}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([r["terrain_kind"] for r in results],
           [r["rmse_after"] for r in results],
           color=["#bbb", "#88a", "#557"])
    ax.set_ylabel("RMSE*, м")
    ax.set_title("Точность TRN по типам рельефа")
    for i, r in enumerate(results):
        ax.text(i, r["rmse_after"], f"{r['rmse_after']:.2f}",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 82)
    print(f"{'Рельеф':<15}{'RMSE*':>10}{'final':>10}"
          f"{'conv_abs':>10}{'conv_rel':>10}{'conv_stable':>12}")
    print("-" * 82)
    for r in results:
        print(f"{r['terrain_kind']:<15}"
              f"{r['rmse_after']:>10.2f}"
              f"{r['final_err']:>10.2f}"
              f"{str(r['convergence_step']):>10}"
              f"{str(r['convergence_step_relative']):>10}"
              f"{str(r['convergence_step_stable']):>12}")
    print("=" * 82)


if __name__ == "__main__":
    main()
