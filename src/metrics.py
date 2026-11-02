from __future__ import annotations

import numpy as np


def position_errors(est_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(est_xy - true_xy, axis=1)


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt((errors ** 2).mean()))


def rmse_after(errors: np.ndarray, burn_in: int) -> float:
    k = min(burn_in, len(errors) - 1)
    return float(np.sqrt((errors[k:] ** 2).mean()))


def max_error(errors: np.ndarray) -> float:
    return float(errors.max())


def final_error(errors: np.ndarray) -> float:
    return float(errors[-1])


def convergence_time(errors: np.ndarray, threshold: float,
                     hold_steps: int = 5) -> int | None:
    below = errors < threshold
    if hold_steps > len(errors):
        return None
    kernel = np.ones(hold_steps, dtype=int)
    counts = np.convolve(below.astype(int), kernel, mode="valid")
    idx = np.where(counts == hold_steps)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def convergence_time_relative(errors: np.ndarray, factor: float = 2.0,
                              hold_steps: int = 5) -> int | None:
    if len(errors) < hold_steps + 1:
        return None
    final = errors[-hold_steps:].mean()
    if final <= 0:
        return None
    threshold = factor * final
    return convergence_time(errors, threshold, hold_steps)


def convergence_time_stable(errors: np.ndarray, window: int = 10,
                            rtol: float = 0.1) -> int | None:
    n = len(errors)
    if n < 2 * window:
        return None
    for i in range(window, n - window):
        before = errors[i - window:i].mean()
        after = errors[i:i + window].mean()
        if before > 0 and (before - after) / before < rtol:
            return i
    return None


def summarize(errors: np.ndarray, burn_in: int = 10,
              convergence_threshold: float = 2.0) -> dict:
    return {
        "rmse": rmse(errors),
        "rmse_after": rmse_after(errors, burn_in),
        "max": max_error(errors),
        "final": final_error(errors),
        "convergence_step": convergence_time(errors, convergence_threshold),
        "convergence_step_relative": convergence_time_relative(errors),
        "convergence_step_stable": convergence_time_stable(errors),
    }
