from __future__ import annotations
import numpy as np

from .terrain_map import TerrainMap


class TRNParticleFilter:
    def __init__(
        self,
        terrain: TerrainMap,
        n_particles: int = 2000,
        init_xy: tuple[float, float] = (0.0, 0.0),
        init_sigma_xy: float = 50.0,
        sigma_meas: float = 3.0,
        sigma_proc: float = 1.0,
        resample_threshold: float = 0.5,
        seed: int | None = None,
    ):
        self.terrain = terrain
        self.n = n_particles
        self.sigma_meas = float(sigma_meas)
        self.sigma_proc = float(sigma_proc)
        self.resample_threshold = float(resample_threshold)
        self.last_gated = False

        rng = np.random.default_rng(seed)
        self.rng = rng

        self.particles = rng.normal(
            loc=np.array(init_xy),
            scale=init_sigma_xy,
            size=(n_particles, 2),
        ).astype(np.float64)

        self.weights = np.ones(n_particles, dtype=np.float64) / n_particles

    def predict(self, velocity_xy, dt: float = 1.0):
        velocity_xy = np.asarray(velocity_xy, dtype=np.float64)
        noise = self.rng.normal(0.0, self.sigma_proc, size=(self.n, 2))
        self.particles += velocity_xy * dt + noise

        self.particles[:, 0] = np.clip(self.particles[:, 0],
                                       self.terrain.x_min, self.terrain.x_max)
        self.particles[:, 1] = np.clip(self.particles[:, 1],
                                       self.terrain.y_min, self.terrain.y_max)

    def update(self, measured_height: float, eps: float = 1e-6,
               gate_threshold: float = 1e-3):
        predicted = self.terrain.height_at_many(
            self.particles[:, 0], self.particles[:, 1], order=1
        )
        errors = measured_height - predicted
        valid = np.isfinite(errors)

        sigma_eff = max(self.sigma_meas, eps)
        likelihood = np.zeros(self.n, dtype=np.float64)
        likelihood[valid] = np.exp(-(errors[valid] ** 2) / (2 * sigma_eff ** 2))

        if likelihood.max() < gate_threshold:
            self.last_gated = True
            return

        self.last_gated = False

        self.weights *= likelihood
        s = self.weights.sum()
        if s <= 0 or not np.isfinite(s):
            self.weights[:] = 1.0 / self.n
        else:
            self.weights /= s

    def neff(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def maybe_resample(self):
        if self.neff() < self.resample_threshold * self.n:
            self.resample()

    def resample(self):
        n = self.n
        positions = (self.rng.random() + np.arange(n)) / n
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        indexes = np.zeros(n, dtype=int)
        i, j = 0, 0
        while i < n:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights[:] = 1.0 / n

    def estimate(self) -> np.ndarray:
        return np.average(self.particles, weights=self.weights, axis=0)

    def estimate_cov(self) -> np.ndarray:
        mean = self.estimate()
        d = self.particles - mean
        return (self.weights[:, None] * d).T @ d

    def weights_summary(self) -> dict:
        return {
            "n": self.n,
            "neff": float(self.neff()),
            "max_weight": float(self.weights.max()),
            "min_weight": float(self.weights.min()),
            "n_zero": int((self.weights < 1e-12).sum()),
        }
