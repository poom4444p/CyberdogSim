"""Tests for affordance/runtime.py: the elevation memory and the walk-ahead helper.

No simulator and no trained weights: hand-made points in, patches out.
"""
import math

import numpy as np
import pytest

from cyberdog.affordance import data
from cyberdog.affordance import runtime as R


def _floor(x0, x1, y0, y1, z=0.0, step=0.05):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


class TestElevationMemory:
    def test_a_flat_floor_reads_level(self):
        m = R.ElevationMemory()
        m.add(_floor(-1, 2, -1.2, 1.2), (0.0, 0.0), 0.0)
        p = m.patch((0.0, 0.0, 0.0))
        assert np.isfinite(p).mean() > 0.9
        assert np.nanmax(np.abs(p)) < 1e-6

    def test_the_ceiling_is_not_ground(self):
        # Highest-wins with a ceiling in it turned every corridor into a wall.
        m = R.ElevationMemory()
        m.add(np.vstack([_floor(-1, 2, -1.2, 1.2), _floor(-1, 2, -1.2, 1.2, z=2.1)]),
              (0.0, 0.0), 0.0)
        assert np.nanmax(m.patch((0.0, 0.0, 0.0))) < 1e-6

    def test_a_person_who_walked_on_leaves_no_ghost(self):
        # Seen standing 1 m ahead, then the floor there is seen again, empty.
        m = R.ElevationMemory()
        person = _floor(0.9, 1.1, -0.1, 0.1, z=0.8)
        m.add(np.vstack([_floor(-1, 2, -1.2, 1.2), person]), (0.0, 0.0), 0.0)
        assert np.nanmax(m.patch((0.0, 0.0, 0.0))) > 0.5
        m.add(_floor(-1, 2, -1.2, 1.2), (0.05, 0.0), 0.0)
        assert np.nanmax(m.patch((0.05, 0.0, 0.0))) < 1e-6

    def test_what_was_not_seen_again_is_remembered(self):
        # A crate seen from back there is still there when the scan misses it.
        m = R.ElevationMemory()
        crate = _floor(1.0, 1.4, -0.2, 0.2, z=0.5)
        m.add(np.vstack([_floor(-1, 2, -1.2, 1.2), crate]), (0.0, 0.0), 0.0)
        m.add(_floor(-1, 0.8, -1.2, 1.2), (0.1, 0.0), 0.0)
        assert np.nanmax(m.patch((0.1, 0.0, 0.0))) == pytest.approx(0.5)

    def test_it_forgets_after_keep_m_of_walking(self):
        m = R.ElevationMemory(keep_m=1.0)
        m.add(_floor(0.9, 1.1, -0.1, 0.1, z=0.3), (0.0, 0.0), 0.0)
        for x in np.arange(0.1, 1.3, 0.1):
            m.add(np.empty((0, 3)), (x, 0.0), 0.0)
        assert not np.isfinite(m.patch((1.2, 0.0, 0.0))).any()

    def test_a_new_floor_starts_empty(self):
        m = R.ElevationMemory()
        m.add(_floor(-1, 2, -1.2, 1.2), (0.0, 0.0), 0.0)
        m.add(np.empty((0, 3)), (0.0, 0.0), 2.2)
        assert not np.isfinite(m.patch((0.0, 0.0, 0.0))).any()


class TestMappedWalls:
    # A wall from y = 0.6, 0.9 m tall; the map gives metres to its face, 0 inside.
    @staticmethod
    def _wall_at(x, y):
        return max(0.6 - y, 0.0)

    def _memory(self, extra=()):
        m = R.ElevationMemory()
        wall = _floor(-1, 2, 0.6, 0.7, z=0.9)
        m.add(np.vstack([_floor(-1, 2, -1.2, 0.6), wall, *extra]), (0.0, 0.0), 0.0)
        return m

    def test_without_the_map_the_wall_is_there(self):
        assert np.nanmax(self._memory().patch((0.0, 0.0, 0.0))) == pytest.approx(0.9)

    def test_a_mapped_wall_reads_as_floor(self):
        p = self._memory().patch((0.0, 0.0, 0.0), self._wall_at)
        assert np.nanmax(p) < 1e-6

    def test_a_crate_by_the_wall_is_still_seen(self):
        # The map knows the wall, not the crate 0.3 m off it.
        crate = _floor(0.8, 1.2, 0.0, 0.3, z=0.5)
        p = self._memory([crate]).patch((0.0, 0.0, 0.0), self._wall_at)
        assert np.nanmax(p) == pytest.approx(0.5)


class TestAhead:
    def test_straight_ahead_is_capped_at_reach(self):
        assert R.ahead((0, 0, 0), (5.0, 0.0)) == pytest.approx((1.2, 0.0))

    def test_too_near_is_none(self):
        assert R.ahead((0, 0, 0), (0.3, 0.0)) is None

    def test_behind_or_turning_is_none(self):
        assert R.ahead((0, 0, 0), (-2.0, 0.0)) is None
        assert R.ahead((0, 0, 0), (1.0, 2.0)) is None

    def test_it_is_in_the_dogs_frame(self):
        # Facing north: a point 2 m north is straight ahead.
        dx, dy = R.ahead((0, 0, math.pi / 2), (0.0, 2.0))
        assert dx == pytest.approx(1.2) and dy == pytest.approx(0.0, abs=1e-9)
