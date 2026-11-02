from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Trajectory:
    times: np.ndarray
    positions_xy: np.ndarray
    velocities_xy: np.ndarray
    headings_deg: np.ndarray

    @classmethod
    def from_straight(
        cls,
        start_xy: Sequence[float],
        speed: float,
        heading_deg: float,
        duration: float,
        dt: float,
    ) -> "Trajectory":
        n = int(round(duration / dt)) + 1
        t = np.linspace(0.0, duration, n)

        heading_rad = np.deg2rad(heading_deg)
        vx = speed * np.cos(heading_rad)
        vy = speed * np.sin(heading_rad)

        x0, y0 = start_xy
        x = x0 + vx * t
        y = y0 + vy * t

        positions = np.column_stack([x, y])
        velocities = np.tile([vx, vy], (n, 1))
        headings = np.full(n, heading_deg, dtype=float)

        return cls(times=t, positions_xy=positions,
                   velocities_xy=velocities, headings_deg=headings)

    @classmethod
    def from_waypoints(
        cls,
        waypoints_xy: Sequence[Sequence[float]],
        speed: float,
        dt: float,
    ) -> "Trajectory":
        wp = np.asarray(waypoints_xy, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2 or wp.shape[0] < 2:
            raise ValueError("waypoints_xy должен быть (M>=2, 2)")

        seg_vec = np.diff(wp, axis=0)
        seg_len = np.linalg.norm(seg_vec, axis=1)
        seg_time = seg_len / speed
        total_time = seg_time.sum()

        n = int(round(total_time / dt)) + 1
        t = np.linspace(0.0, total_time, n)

        cum_t = np.concatenate([[0.0], np.cumsum(seg_time)])

        x = np.interp(t, cum_t, wp[:, 0])
        y = np.interp(t, cum_t, wp[:, 1])
        positions = np.column_stack([x, y])

        vx = np.gradient(x, t, edge_order=2)
        vy = np.gradient(y, t, edge_order=2)
        velocities = np.column_stack([vx, vy])

        headings = np.rad2deg(np.arctan2(vy, vx))

        return cls(times=t, positions_xy=positions,
                   velocities_xy=velocities, headings_deg=headings)

    def __len__(self) -> int:
        return self.times.size

    def true_heights(self, terrain) -> np.ndarray:
        return terrain.height_at_many(
            self.positions_xy[:, 0], self.positions_xy[:, 1], order=1
        )

    def subset(self, start: int, stop: int) -> "Trajectory":
        return Trajectory(
            times=self.times[start:stop],
            positions_xy=self.positions_xy[start:stop],
            velocities_xy=self.velocities_xy[start:stop],
            headings_deg=self.headings_deg[start:stop],
        )
