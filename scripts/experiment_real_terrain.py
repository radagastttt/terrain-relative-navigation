from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scripts.run_experiment import build_terrain, run_once
from src.trajectory import Trajectory
from src.utils import load_config, set_seed, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REGIONS = {
    "flat": ("Горнотракийская низменность",
             "data/raw/thracian_plain_cop30_utm35n.tif"),
    "hilly": ("Предбалкан (Ловеч)",
              "data/raw/pre_balkan_lovech_cop30_utm35n.tif"),
    "mountainous": ("Централен Балкан (Ботев)",
                    "data/raw/central_balkan_cop30_utm35n.tif"),
}
COLORS = {"flat": "tab:olive", "hilly": "tab:orange", "mountainous": "tab:purple"}


def region_cfg(cfg: dict, path: str) -> dict:
    c = copy.deepcopy(cfg)
    c["dem"]["source"] = "geotiff"
    c["dem"]["path"] = path
    return c


def terrain_stats(terrain) -> dict:
    dem = terrain.dem
    gy, gx = np.gradient(dem, terrain.dy, terrain.dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return {
        "h_min": float(dem.min()), "h_max": float(dem.max()),
        "h_std": float(dem.std()), "slope_mean": float(slope.mean()),
    }


def random_track(terrain, tr: dict, rng, margin: float):
    x0, x1, y0, y1 = terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max
    length = tr["speed"] * tr["duration"]
    for _ in range(1000):
        start = rng.uniform([x0 + margin, y0 + margin], [x1 - margin, y1 - margin])
        heading = rng.uniform(0.0, 360.0)
        a = np.radians(heading)
        end = start + length * np.array([np.cos(a), np.sin(a)])
        if x0 + margin <= end[0] <= x1 - margin and y0 + margin <= end[1] <= y1 - margin:
            return start, heading
    raise RuntimeError("Не удалось подобрать траекторию внутри DEM.")


def run_region(cfg, kind: str, logger, args) -> dict:
    name, path = REGIONS[kind]
    rcfg = region_cfg(cfg, path)
    terrain = build_terrain(rcfg)
    stats = terrain_stats(terrain)
    logger.info(f"[{kind}] {name}: h={stats['h_min']:.0f}..{stats['h_max']:.0f} м  "
                f"σh={stats['h_std']:.1f} м  уклон={stats['slope_mean']:.1f}°")

    rng = np.random.default_rng(cfg["seed"])
    tr = rcfg["trajectory"]
    runs = []
    for k in range(args.runs):
        start, heading = random_track(terrain, tr, rng, args.margin)
        offset = rng.normal(0.0, args.sigma0, size=2)
        rcfg["trajectory"]["start_xy"] = [float(start[0]), float(start[1])]
        rcfg["trajectory"]["heading_deg"] = float(heading)

        traj = Trajectory.from_straight(start, tr["speed"], heading,
                                        tr["duration"], tr["dt"])
        prof = traj.true_heights(terrain)

        res = run_once(rcfg, logger, sigma_meas=args.sigma_meas, sigma_proc=0.1,
                       n_particles=args.n_particles, seed_offset=k,
                       tag=f"{kind} run={k}", burn_in=args.burn_in,
                       init_sigma_xy=args.sigma0, init_offset_xy=tuple(offset))
        runs.append({
            "start": start, "heading": heading,
            "positions": traj.positions_xy,
            "profile_std": float(prof.std()),
            "profile_dh": float(np.mean(np.abs(np.diff(prof)))),
            "rmse_after": res["rmse_after"], "final": res["final_err"],
        })

    final = np.array([r["final"] for r in runs])
    rmse_after = np.array([r["rmse_after"] for r in runs])
    out = {
        "kind": kind, "name": name, "terrain": terrain, "stats": stats, "runs": runs,
        "success_rate": float(np.mean(final < args.success_radius)),
        "rmse_med": float(np.median(rmse_after)),
        "final_med": float(np.median(final)),
        "final_p90": float(np.percentile(final, 90)),
        "profile_std_med": float(np.median([r["profile_std"] for r in runs])),
    }
    logger.info(f"=== [{kind}] success={100 * out['success_rate']:.0f}%  "
                f"RMSE* med={out['rmse_med']:.2f} м  final med={out['final_med']:.2f} м  "
                f"final p90={out['final_p90']:.2f} м  σ профиля мед.={out['profile_std_med']:.1f} м")
    return out


def plot_maps(results: list[dict], out_path: Path, success_radius: float):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5))
    for ax, r in zip(np.atleast_1d(axes), results):
        t = r["terrain"]
        extent = [t.x_min / 1e3, t.x_max / 1e3, t.y_min / 1e3, t.y_max / 1e3]
        im = ax.imshow(t.dem, origin="lower", extent=extent, cmap="terrain")
        for run in r["runs"]:
            p = run["positions"] / 1e3
            ok = run["final"] < success_radius
            ax.plot(p[:, 0], p[:, 1], color="k" if ok else "red", lw=1.2)
            ax.plot(p[0, 0], p[0, 1], ".", color="k" if ok else "red", ms=4)
        ax.set_title(f"{r['name']}\nуспех {100 * r['success_rate']:.0f}%  "
                     f"(чёрные — успех, красные — отказ)", fontsize=10)
        ax.set_xlabel("E, км (UTM 35N)")
        ax.set_ylabel("N, км")
        fig.colorbar(im, ax=ax, label="Высота, м", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(results: list[dict], out_path: Path, success_radius: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        x = [run["profile_std"] for run in r["runs"]]
        y = [max(run["final"], 1e-2) for run in r["runs"]]
        axes[0].scatter(x, y, s=25, color=COLORS[r["kind"]], label=r["name"], alpha=0.8)
    axes[0].axhline(success_radius, color="k", ls=":", label=f"порог {success_radius:g} м")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("СКО высоты вдоль траектории, м")
    axes[0].set_ylabel("Финальная ошибка, м")
    axes[0].set_title("Ошибка привязки vs информативность профиля")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    labels = [r["kind"] for r in results]
    rate = [100 * r["success_rate"] for r in results]
    bars = axes[1].bar(labels, rate, color=[COLORS[k] for k in labels])
    for b, r in zip(bars, results):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                     f"{b.get_height():.0f}%\nfinal мед. {r['final_med']:.1f} м",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 120)
    axes[1].set_ylabel("Успешные привязки, %")
    axes[1].set_title(f"Доля прогонов с финальной ошибкой < {success_radius:g} м")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=100.0)
    parser.add_argument("--sigma-meas", type=float, default=1.0)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--margin", type=float, default=1000.0)
    parser.add_argument("--success-radius", type=float, default=30.0)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "figures" / "real_terrain"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    args.n_particles = args.n_particles or cfg["particle_filter"]["n_particles"]
    logger = setup_logger("trn_real_terrain",
                          str(PROJECT_ROOT / "results" / "logs" / "real_terrain.log"))
    logger.info(f"runs={args.runs}  σ0={args.sigma0}  σ_meas={args.sigma_meas}  "
                f"n_particles={args.n_particles}")

    results = [run_region(cfg, k, logger, args) for k in args.regions]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_maps(results, out_dir / "real_terrain_maps.png", args.success_radius)
    plot_summary(results, out_dir / "real_terrain_summary.png", args.success_radius)

    print("\n" + "=" * 100)
    print(f"{'Рельеф':<13}{'высоты, м':>12}{'σh, м':>8}{'уклон, °':>10}"
          f"{'σ проф., м':>12}{'успех, %':>10}{'RMSE* мед.':>12}"
          f"{'final мед.':>12}{'final p90':>11}")
    print("-" * 100)
    for r in results:
        s = r["stats"]
        print(f"{r['kind']:<13}{s['h_min']:>5.0f}..{s['h_max']:<5.0f}"
              f"{s['h_std']:>8.1f}{s['slope_mean']:>10.1f}"
              f"{r['profile_std_med']:>12.1f}{100 * r['success_rate']:>10.0f}"
              f"{r['rmse_med']:>12.2f}{r['final_med']:>12.2f}{r['final_p90']:>11.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
