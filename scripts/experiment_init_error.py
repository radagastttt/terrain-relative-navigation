from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scripts.run_experiment import run_once
from src.utils import load_config, set_seed, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SIGMAS = [50, 100, 200, 300, 500, 750, 1000]


def run_series(cfg, logger, sigma0: float, n_runs: int, sigma_meas: float,
               n_particles: int, burn_in: int, success_radius: float) -> dict:
    rng = np.random.default_rng(cfg["seed"] + int(sigma0))
    rmse_after, final, init_err, conv = [], [], [], []
    errors = []
    for k in range(n_runs):
        offset = rng.normal(0.0, sigma0, size=2)
        res = run_once(
            cfg, logger,
            sigma_meas=sigma_meas, sigma_proc=0.1,
            n_particles=n_particles, seed_offset=1000 * k + int(sigma0),
            tag=f"s0={sigma0:g} run={k}", burn_in=burn_in,
            init_sigma_xy=sigma0, init_offset_xy=tuple(offset),
        )
        init_err.append(float(np.linalg.norm(offset)))
        rmse_after.append(res["rmse_after"])
        final.append(res["final_err"])
        errors.append(res["errors"])
        above = np.nonzero(res["errors"] >= success_radius)[0]
        step = 0 if len(above) == 0 else int(above[-1]) + 1
        conv.append(step if step < len(res["errors"]) else None)

    final = np.array(final)
    rmse_after = np.array(rmse_after)
    conv_ok = [c for c in conv if c is not None]
    return {
        "sigma0": sigma0,
        "init_err_mean": float(np.mean(init_err)),
        "rmse_after": rmse_after,
        "final": final,
        "success_rate": float(np.mean(final < success_radius)),
        "conv_median": float(np.median(conv_ok)) if conv_ok else None,
        "errors": np.array(errors),
        "example": res,
    }


def plot_summary(series: list[dict], out_path: Path, success_radius: float):
    s0 = np.array([s["sigma0"] for s in series])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, label, color in (("rmse_after", "RMSE*", "tab:blue"),
                              ("final", "Финальная ошибка", "tab:red")):
        med = np.array([np.median(s[key]) for s in series])
        q1 = np.array([np.percentile(s[key], 25) for s in series])
        q3 = np.array([np.percentile(s[key], 75) for s in series])
        axes[0].plot(s0, med, "o-", color=color, label=f"{label} (медиана)")
        axes[0].fill_between(s0, q1, q3, color=color, alpha=0.2)
    axes[0].axhline(success_radius, color="k", ls=":",
                    label=f"порог успеха {success_radius:g} м")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Начальная ошибка σ0, м")
    axes[0].set_ylabel("Ошибка, м")
    axes[0].set_title("Точность привязки (медиана, IQR)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    rate = np.array([100 * s["success_rate"] for s in series])
    axes[1].plot(s0, rate, "o-", color="tab:green")
    for x, y in zip(s0, rate):
        axes[1].annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center")
    axes[1].set_ylim(-5, 110)
    axes[1].set_xlabel("Начальная ошибка σ0, м")
    axes[1].set_ylabel("Успешные привязки, %")
    axes[1].set_title(f"Доля прогонов с финальной ошибкой < {success_radius:g} м")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_curves(series: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in series:
        ax.plot(np.median(s["errors"], axis=0), label=f"σ0={s['sigma0']:g} м")
    ax.set_yscale("log")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Ошибка, м (медиана по прогонам)")
    ax.set_title("Сходимость TRN при разной начальной ошибке")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sigma-meas", type=float, default=1.0)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--success-radius", type=float, default=30.0)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "figures" / "init_error"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = setup_logger("trn_init_error",
                          str(PROJECT_ROOT / "results" / "logs" / "init_error.log"))
    n_particles = args.n_particles or cfg["particle_filter"]["n_particles"]
    out_dir = Path(args.out)

    logger.info(f"DEM={cfg['dem']['source']}  runs={args.runs}  "
                f"σ_meas={args.sigma_meas}  n_particles={n_particles}")
    series = []
    for s0 in args.sigmas:
        s = run_series(cfg, logger, s0, args.runs, args.sigma_meas,
                       n_particles, args.burn_in, args.success_radius)
        logger.info(
            f"=== σ0={s0:g} м: success={100 * s['success_rate']:.0f}%  "
            f"RMSE* med={np.median(s['rmse_after']):.2f} м  "
            f"final med={np.median(s['final']):.2f} м  conv med={s['conv_median']}"
        )
        series.append(s)

    plot_summary(series, out_dir / "init_error_summary.png", args.success_radius)
    plot_error_curves(series, out_dir / "init_error_convergence.png")

    print("\n" + "=" * 92)
    print(f"{'σ0, м':>8}{'|Δ0| ср., м':>14}{'успех, %':>10}"
          f"{'RMSE* мед.':>12}{'RMSE* p90':>11}{'final мед.':>12}"
          f"{'final p90':>11}{'сход., шаг':>12}")
    print("-" * 92)
    for s in series:
        conv = "—" if s["conv_median"] is None else f"{s['conv_median']:.0f}"
        print(f"{s['sigma0']:>8g}{s['init_err_mean']:>14.0f}"
              f"{100 * s['success_rate']:>10.0f}"
              f"{np.median(s['rmse_after']):>12.2f}"
              f"{np.percentile(s['rmse_after'], 90):>11.2f}"
              f"{np.median(s['final']):>12.2f}"
              f"{np.percentile(s['final'], 90):>11.2f}"
              f"{conv:>12}")
    print("=" * 92)


if __name__ == "__main__":
    main()
