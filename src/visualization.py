from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.terrain_map import TerrainMap


def plot_trajectory(res: dict, out_path: Path):
    terrain: TerrainMap = res["terrain"]
    true = res["history_true"]
    est = res["history_est"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(
        terrain.dem, cmap="terrain", origin="lower",
        extent=[terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max],
    )
    axes[0].plot(true[:, 0], true[:, 1], "r--", lw=1.5, label="True")
    axes[0].plot(est[:, 0], est[:, 1], "b-", lw=1.5, label="TRN estimate")
    axes[0].plot(true[0, 0], true[0, 1], "go", ms=8, label="start")
    axes[0].set_title(f"Траектория ({res['tag']})")
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("x, м")
    axes[0].set_ylabel("y, м")

    axes[1].plot(res["errors"], "k-", label="Ошибка")
    axes[1].axhline(res["rmse_after"], color="b", ls=":",
                    label=f"RMSE*={res['rmse_after']:.2f} м")
    axes[1].set_xlabel("Шаг")
    axes[1].set_ylabel("Ошибка, м")
    axes[1].set_title(f"Ошибка позиции (RMSE={res['rmse']:.2f} м)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_errors_comparison(results: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in results:
        ax.plot(r["errors"], label=f"{r['tag']} (RMSE*={r['rmse_after']:.2f})")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Ошибка, м")
    ax.set_title("Сравнение ошибок по стадиям")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_neff(result: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["neff"], "b-")
    ax.axhline(result["neff"].mean(), color="r", ls="--",
               label=f"mean={result['neff'].mean():.0f}")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Neff")
    ax.set_title(f"Динамика Neff ({result['tag']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_experiment_summary(experiments: list[dict], out_path: Path,
                            xlabel: str = "Параметр"):
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [e["x"] for e in experiments]
    ys = [e["rmse_after"] for e in experiments]
    ax.plot(xs, ys, "o-", lw=2, ms=8)
    for e in experiments:
        ax.annotate(f"{e['rmse_after']:.2f}",
                    (e["x"], e["rmse_after"]),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE*, м")
    ax.set_title(f"Зависимость RMSE* от {xlabel}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
