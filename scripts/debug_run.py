from pathlib import Path
import numpy as np

from src.terrain_map import TerrainMap, make_synthetic_dem
from src.trajectory import Trajectory
from src.measurement import Barometer
from src.particle_filter import TRNParticleFilter
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def one_run(sigma_meas, sigma_proc, n_particles, init_sigma_xy,
            bias, outlier_prob, seed=42):
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    dem = make_synthetic_dem(cfg["dem"]["rows"], cfg["dem"]["cols"], seed=cfg["seed"])
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
    true_h = traj.true_heights(terrain)

    baro = Barometer(sigma=sigma_meas, bias=bias,
                     outlier_prob=outlier_prob, bias_sigma=0.0, seed=seed)
    measured = baro.measure_many(true_h)

    pf = TRNParticleFilter(
        terrain, n_particles=n_particles,
        init_xy=tr["start_xy"], init_sigma_xy=init_sigma_xy,
        sigma_meas=sigma_meas, sigma_proc=sigma_proc, seed=seed,
    )

    errs = []
    for i in range(len(traj)):
        pf.predict(traj.velocities_xy[i], tr["dt"])
        pf.update(float(measured[i]))
        pf.maybe_resample()
        errs.append(np.linalg.norm(pf.estimate() - traj.positions_xy[i]))
    errs = np.array(errs)
    return float(np.sqrt((errs[10:] ** 2).mean())), float(errs[-1])


print(f"{'Сценарий':<50} {'RMSE*':>8} {'final':>8}")
print("-" * 68)

scenarios = [
    ("σ=0.5 bias=0.0 out=0.00 (идеально)",     0.5, 0.5, 5000, 30, 0.0, 0.0),
    ("σ=0.5 bias=0.5 out=0.00 (только bias)",  0.5, 0.5, 5000, 30, 0.5, 0.0),
    ("σ=0.5 bias=0.0 out=0.02 (только выбросы)", 0.5, 0.5, 5000, 30, 0.0, 0.02),
    ("σ=0.5 bias=0.5 out=0.02 (как в run_experiment)", 0.5, 0.5, 5000, 30, 0.5, 0.02),
    ("σ=3.0 bias=0.0 out=0.00",                 3.0, 0.5, 5000, 30, 0.0, 0.0),
    ("σ=3.0 bias=0.5 out=0.02",                 3.0, 0.5, 5000, 30, 0.5, 0.02),
]

for name, sm, sp, n, isig, bias, op in scenarios:
    rmse, final = one_run(sm, sp, n, isig, bias, op)
    print(f"{name:<50} {rmse:>8.2f} {final:>8.2f}")
