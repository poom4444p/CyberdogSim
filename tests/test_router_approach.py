"""Tests for BuildingRouter.approach() -- where the dog stands by a no-go zone.

The regression these guard: "stairs" in locations.json is a point in the middle
of a solid block of occupied cells, and nearest_free's spread only ever stepped
onto free cells. From an occupied start it had nowhere to step, returned the
stairs point itself, and every "take me to the stairs" ended as `no route to
stairs` -- the refusal the spec promises, never spoken, on all three floors.
"""
import math

import pytest


from cyberdog.planning.building_router import APPROACH_CLEARANCE, BuildingRouter


@pytest.fixture(scope="module")
def router():
    return BuildingRouter()


class TestStairsApproach:
    """Asking for the stairs: refused, but walked to and said out loud."""

    @pytest.mark.parametrize("floor", (1, 2, 3))
    def test_approach_is_outside_the_stairwell(self, router, floor):
        target, line = router.approach("stairs", floor, (45.0, 9.5), floor=floor)
        xy = tuple(target["xy"])
        assert target["floor"] == floor
        assert "no closer" in line
        # Not the stairs point itself -- that is the bug, and it is inside the
        # zone the dog is being kept out of.
        assert math.dist(xy, (1.0, 9.5)) > APPROACH_CLEARANCE
        assert router.clear_of_zones(xy, APPROACH_CLEARANCE)

    @pytest.mark.parametrize("floor", (1, 2, 3))
    def test_approach_is_somewhere_a_route_can_end(self, router, floor):
        target, _ = router.approach("stairs", floor, (45.0, 9.5), floor=floor)
        grid = router.plan_grids[floor]
        assert grid.is_free(*grid.world_to_grid(*target["xy"]))
        # And reachable: the lift is at the other end of the same corridor.
        assert router.plan(floor, (46.6, 9.5), target) is not None

    def test_unknown_name_is_none(self, router):
        assert router.approach("teleporter", 1, (45.0, 9.5)) is None


class TestNearestFree:
    """The spread itself: out of the blob it starts in, but not through walls."""

    def test_free_start_is_returned_unchanged(self, router):
        # A point in open corridor, well clear of every zone, is its own answer.
        assert router.nearest_free(1, (20.0, 9.5), clearance=APPROACH_CLEARANCE) \
            == pytest.approx((20.0, 9.5), abs=0.1)

    def test_occupied_start_escapes_the_zone(self, router):
        xy = router.nearest_free(1, (1.0, 9.5), clearance=APPROACH_CLEARANCE)
        assert router.clear_of_zones(xy, APPROACH_CLEARANCE)
        # Out the corridor side (east), not through the wall into the room
        # next door: the stairwell's only opening is the one the dog came from.
        assert xy[0] > 2.5 and 8.3 < xy[1] < 10.7
