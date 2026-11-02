from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Barometer:
    sigma: float = 3.0
    bias: float = 0.0
    outlier_prob: float = 0.0
    outlier_scale: float = 5.0
    bias_ar1: float = 0.99
    bias_sigma: float = 0.1
    seed: int | None = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)

    def measure(self, true_height: float, state: dict | None = None) -> tuple[float, dict]:
        if state is None:
            state = {"bias": self.bias}

        state["bias"] = (
            self.bias_ar1 * state["bias"]
            + self._rng.normal(0.0, self.bias_sigma)
        )

        if self.outlier_prob > 0 and self._rng.random() < self.outlier_prob:
            e = self._rng.normal(0.0, self.outlier_scale * self.sigma)
        else:
            e = self._rng.normal(0.0, self.sigma)

        z = true_height + state["bias"] + e
        return float(z), state

    def measure_many(self, true_heights: np.ndarray) -> np.ndarray:
        n = len(true_heights)

        bias_series = np.empty(n, dtype=float)
        b = self.bias
        for i in range(n):
            b = self.bias_ar1 * b + self._rng.normal(0.0, self.bias_sigma)
            bias_series[i] = b

        noise = self._rng.normal(0.0, self.sigma, size=n)
        if self.outlier_prob > 0:
            mask = self._rng.random(n) < self.outlier_prob
            noise[mask] = self._rng.normal(
                0.0, self.outlier_scale * self.sigma, size=mask.sum()
            )

        return (true_heights + bias_series + noise).astype(np.float32)
