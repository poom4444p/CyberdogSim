"""Tests for affordance/data.py: the patch, the label, the shard format. No Isaac needed."""

import math

import numpy as np
import pytest

from cyberdog.affordance import data


def _record(n=3, **overrides):
    r = {
        "patch": np.zeros((n, data.NY, data.NX), np.float32),
        "target": np.zeros((n, 2), np.float32),
        "reached": np.ones(n, bool),
        "fell": np.zeros(n, bool),
        "duration": np.ones(n, np.float32),
        "max_tilt": np.zeros(n, np.float32),
        "dz": np.zeros(n, np.float32),
        "z_span": np.zeros(n, np.float32),
        "terrain_type": np.zeros(n, np.int16),
        "terrain_level": np.zeros(n, np.int16),
    }
    r.update(overrides)
    return r


def _cell(x, y):
    """(row, col) of the cell centred on (x, y) in the dog's frame."""
    return int(round((y - data.Y0) / data.RES)), int(round((x - data.X0) / data.RES))


class TestPatch:

    def test_flat_ground_reads_zero_everywhere_it_was_seen(self):
        xs, ys = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(-2, 2, 0.05))
        pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 3.0)], axis=1)
        p = data.patch_from_points(pts, (0.0, 0.0, 0.0), ground_z=3.0)
        assert p.shape == (data.NY, data.NX)
        assert np.all(p == 0.0)

    def test_a_point_lands_in_the_cell_ahead_when_the_dog_faces_it(self):
        # Dog at (5, 5) facing +y: a point 1 m north is 1 m *ahead*.
        p = data.patch_from_points([[5.0, 6.0, 0.3]], (5.0, 5.0, math.pi / 2), 0.0)
        assert p[_cell(1.0, 0.0)] == pytest.approx(0.3)
        assert np.isnan(p).sum() == p.size - 1

    def test_left_is_plus_y(self):
        # Facing +x, a point at world +y is on the dog's left.
        p = data.patch_from_points([[0.0, 0.5, 0.1]], (0.0, 0.0, 0.0), 0.0)
        assert p[_cell(0.0, 0.5)] == pytest.approx(0.1)

    def test_a_cell_keeps_its_highest_point(self):
        pts = [[1.0, 0.0, 0.05], [1.02, 0.01, 0.40], [0.98, -0.02, 0.10]]
        p = data.patch_from_points(pts, (0.0, 0.0, 0.0), 0.0)
        assert p[_cell(1.0, 0.0)] == pytest.approx(0.40)

    def test_heights_are_above_the_ground_given(self):
        p = data.patch_from_points([[1.0, 0.0, 10.15]], (0.0, 0.0, 0.0), 10.0)
        assert p[_cell(1.0, 0.0)] == pytest.approx(0.15)

    def test_outside_the_patch_and_missed_rays_are_ignored(self):
        pts = [[5.0, 0.0, 1.0],              # beyond the front edge
               [-1.0, 0.0, 1.0],             # behind the back edge
               [1.0, 0.0, np.inf],           # a ray that hit nothing
               [np.nan, np.nan, np.nan]]
        p = data.patch_from_points(pts, (0.0, 0.0, 0.0), 0.0)
        assert np.isnan(p).all()

    def test_batched_matches_one_at_a_time(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-2, 2, (4, 300, 3))
        poses = rng.uniform(-1, 1, (4, 3))
        gz = rng.uniform(-0.1, 0.1, 4)
        batched = data.patches_from_points(pts, poses, gz)
        for b in range(4):
            np.testing.assert_array_equal(
                batched[b], data.patch_from_points(pts[b], poses[b], gz[b]))

    def test_the_farthest_target_is_inside_the_patch(self):
        d = data.TARGET_D[1]
        xs, ys = data.cell_centres()
        for bearing in (-data.TARGET_BEARING, 0.0, data.TARGET_BEARING):
            x, y = d * math.cos(bearing), d * math.sin(bearing)
            assert xs[0] <= x <= xs[-1]
            assert ys[0] <= y <= ys[-1]


class TestGroundUnder:

    def test_median_of_what_is_under_the_dog(self):
        pts = np.array([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.2], [0.0, 0.1, 1.1],
                         [2.0, 0.0, 9.0]]])                 # far away: ignored
        assert data.ground_under(pts, np.zeros((1, 3)))[0] == pytest.approx(1.1)

    def test_nothing_under_the_dog_is_nan(self):
        pts = np.array([[[2.0, 0.0, 1.0]]])
        assert np.isnan(data.ground_under(pts, np.zeros((1, 3)))[0])


class TestLabel:

    def test_an_easy_walk_is_walkable(self):
        assert data.label(_record(1)).tolist() == [True]

    def test_each_failure_on_its_own_makes_it_unwalkable(self):
        r = _record(4,
                    reached=np.array([False, True, True, True]),
                    fell=np.array([False, True, False, False]),
                    max_tilt=np.array([0, 0, data.TILT_MAX + 0.01, 0], np.float32),
                    z_span=np.array([0, 0, 0, data.STEP_MAX + 0.01], np.float32))
        assert data.label(r).tolist() == [False] * 4

    def test_up_and_back_down_is_still_a_climb(self):
        # Three stairs up and three down: ends level, z_span catches it.
        r = _record(1, dz=np.zeros(1, np.float32),
                    z_span=np.full(1, 0.45, np.float32))
        assert data.label(r).tolist() == [False]

    def test_thresholds_can_be_moved_without_recollecting(self):
        r = _record(1, z_span=np.full(1, 0.10, np.float32))
        assert data.label(r).tolist() == [False]
        assert data.label(r, step_max=0.12).tolist() == [True]

    def test_the_gaits_own_rocking_is_not_a_failure(self):
        # The Go2 rocks 8-10 deg walking on flat ground; that used to fail.
        r = _record(1, max_tilt=np.full(1, math.radians(10), np.float32))
        assert data.label(r).tolist() == [True]


def _step(height, at_x=0.75):
    """A record walking 1.2 m straight ahead over a step `height` high at `at_x`."""
    r = _record(1, target=np.array([[1.2, 0.0]], np.float32))
    xs, _ = data.cell_centres()
    r["patch"][0][:, xs >= at_x] = height
    return r


class TestClasses:

    def test_flat_is_walkable(self):
        assert data.classes(_step(0.0)).tolist() == [data.WALKABLE]

    def test_small_bumps_are_caution(self):
        assert data.classes(_step(0.035)).tolist() == [data.CAUTION]

    def test_an_edge_is_not_walkable(self):
        # A kerb-sized edge: refused even though the dog arrived without falling.
        assert data.classes(_step(0.12)).tolist() == [data.NOT_WALKABLE]

    def test_caution_still_counts_as_followable(self):
        assert data.label(_step(0.035)).tolist() == [True]
        assert data.label(_step(0.12)).tolist() == [False]

    def test_an_edge_beside_the_path_does_not_count(self):
        # The same kerb, but only on cells more than PATH_HALF_W to the left.
        r = _record(1, target=np.array([[1.2, 0.0]], np.float32))
        _, ys = data.cell_centres()
        r["patch"][0][ys > data.PATH_HALF_W + 0.15, :] = 0.12
        assert data.classes(r).tolist() == [data.WALKABLE]

    def test_a_ramp_is_a_slope_not_a_bump(self):
        # 1:12, the steepest ramp an accessibility code allows.
        r = _record(1, target=np.array([[1.2, 0.0]], np.float32))
        xs, _ = data.cell_centres()
        r["patch"][0][:] = np.maximum(xs, 0) / 12.0
        assert data.path_bump(r)[0] < data.BUMP_WALK
        assert data.classes(r).tolist() == [data.WALKABLE]

    def test_unseen_cells_are_not_bumps(self):
        r = _step(0.0)
        r["patch"][0][::2, ::2] = np.nan
        assert data.path_bump(r)[0] == 0.0

    def test_a_failure_wins_over_smooth_ground(self):
        r = _step(0.0)
        r["fell"][:] = True
        assert data.classes(r).tolist() == [data.NOT_WALKABLE]


class TestShards:

    def test_round_trip(self, tmp_path):
        a, b = _record(2), _record(3)
        data.save_shard(tmp_path / "shard_0000.npz", a, {"run": 1})
        data.save_shard(tmp_path / "shard_0001.npz", b, {"run": 1})
        records, metas = data.load_shards(tmp_path)
        assert len(records["patch"]) == 5
        assert metas == [{"run": 1}, {"run": 1}]

    def test_a_missing_field_is_refused(self, tmp_path):
        r = _record(2)
        del r["z_span"]
        with pytest.raises(ValueError, match="z_span"):
            data.save_shard(tmp_path / "x.npz", r, {})

    def test_a_wrong_patch_shape_is_refused(self, tmp_path):
        r = _record(2, patch=np.zeros((2, 5, 5), np.float32))
        with pytest.raises(ValueError, match="patch"):
            data.save_shard(tmp_path / "x.npz", r, {})


def test_walk_speed_is_the_twins():
    """The trials drive at the speed the twin's dog walks, or they measure another dog."""
    mujoco_robot = pytest.importorskip("cyberdog.sim.robot.mujoco_robot")
    assert data.WALK_V == mujoco_robot.MujocoRobot.MAX_V
