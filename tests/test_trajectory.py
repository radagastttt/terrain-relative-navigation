import numpy as np
import pytest

from src.trajectory import Trajectory
from src.terrain_map import TerrainMap, make_synthetic_dem


def test_straight_length():
    traj = Trajectory.from_straight(
        start_xy=(0.0, 0.0), speed=10.0, heading_deg=0.0,
        duration=5.0, dt=1.0,
    )
    assert len(traj) == 6
    np.testing.assert_allclose(traj.times, [0, 1, 2, 3, 4, 5])


def test_straight_positions_east():
    traj = Trajectory.from_straight(
        start_xy=(100.0, 200.0), speed=10.0, heading_deg=0.0,
        duration=5.0, dt=1.0,
    )
    np.testing.assert_allclose(traj.positions_xy[:, 0], [100, 110, 120, 130, 140, 150])
    np.testing.assert_allclose(traj.positions_xy[:, 1], 200.0)


def test_straight_positions_45deg():
    traj = Trajectory.from_straight(
        start_xy=(0.0, 0.0), speed=np.sqrt(2.0), heading_deg=45.0,
        duration=1.0, dt=1.0,
    )
    np.testing.assert_allclose(traj.positions_xy[-1], [1.0, 1.0], atol=1e-9)


def test_true_heights_matches_terrain():
    dem = make_synthetic_dem(rows=100, cols=100, seed=1)
    terrain = TerrainMap(dem, dx=1.0, dy=1.0)

    traj = Trajectory.from_straight(
        start_xy=(10.0, 10.0), speed=5.0, heading_deg=30.0,
        duration=5.0, dt=0.5,
    )
    h_traj = traj.true_heights(terrain)

    h_manual = np.array([
        terrain.height_at(x, y) for x, y in traj.positions_xy
    ])
    np.testing.assert_allclose(h_traj, h_manual, atol=1e-6)


def test_waypoints_pass_through():
    wp = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    traj = Trajectory.from_waypoints(wp, speed=10.0, dt=0.1)

    np.testing.assert_allclose(traj.positions_xy[0], wp[0], atol=1e-6)
    np.testing.assert_allclose(traj.positions_xy[-1], wp[-1], atol=1e-6)

    dist_to_corner = np.linalg.norm(traj.positions_xy - np.array([10.0, 0.0]), axis=1)
    assert dist_to_corner.min() < 1.0
