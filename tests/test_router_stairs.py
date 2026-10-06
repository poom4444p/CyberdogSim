"""Tests for BuildingRouter.choose_transit() -- the lift, or the stairs as a fallback.

Spec rule 0: the lift first; the stairs only when the lift cannot be used, or
is so much further (STAIRS_RATIO times *and* STAIRS_EXTRA_M metres) that the
long way round is the worse service -- and, whichever it is, no route that
merely walks somewhere may cross a stairwell. Asking the person is run_building's
job and is not tested here; deciding what to ask about is.
"""
import pytest

from cyberdog.planning.building_router import (STAIRS_EXTRA_M, STAIRS_RATIO,
                                               BuildingRouter, NoAccessibleRoute)


def trip(router, start, start_floor, dest, dest_floor):
    s = router.resolve(start, start_floor, (0, 0), floor=start_floor)
    t = router.resolve(dest, dest_floor, s["xy"], floor=dest_floor)
    return start_floor, s["xy"], t


@pytest.fixture(scope="module")
def router():
    return BuildingRouter()


@pytest.fixture(scope="module")
def lift_out():
    return BuildingRouter(lift_in_service=False)


@pytest.fixture(scope="module")
def no_stairs():
    return BuildingRouter(allow_stairs=False)


@pytest.fixture(scope="module")
def neither():
    return BuildingRouter(allow_stairs=False, lift_in_service=False)


class TestChoice:
    def test_lift_when_it_is_about_as_near(self, router):
        # From the main entrance the lift is beside you: no reason for steps.
        t = router.choose_transit(*trip(router, "main entrance", 1, "room 201", 2))
        assert t.kind == "lift" and t.reason == ""

    def test_lift_when_it_is_nearer(self, router):
        t = router.choose_transit(*trip(router, "library", 1, "chemistry lab", 3))
        assert t.kind == "lift"
        assert t.lift_m < t.stairs_m

    def test_stairs_when_the_lift_is_much_further(self, router):
        # Both rooms at the west end, by the stairwell; the lift is at the east end.
        t = router.choose_transit(*trip(router, "room 101", 1, "room 201", 2))
        assert t.kind == "stairs"
        assert t.lift_m > STAIRS_RATIO * t.stairs_m
        assert t.lift_m - t.stairs_m > STAIRS_EXTRA_M
        assert "metres" in t.reason

    def test_down_as_well_as_up(self, router):
        t = router.choose_transit(*trip(router, "room 301", 3, "room 101", 1))
        assert t.kind == "stairs"

    def test_stairs_when_the_lift_is_out(self, lift_out):
        t = lift_out.choose_transit(*trip(lift_out, "main entrance", 1, "room 201", 2))
        assert t.kind == "stairs"
        assert t.lift_m is None
        assert "out of service" in t.reason

    def test_never_stairs_when_not_allowed(self, no_stairs):
        t = no_stairs.choose_transit(*trip(no_stairs, "room 101", 1, "room 201", 2))
        assert t.kind == "lift" and t.stairs_m is None

    def test_refused_when_neither(self, neither):
        with pytest.raises(NoAccessibleRoute, match="out of service"):
            neither.choose_transit(*trip(neither, "main entrance", 1, "room 201", 2))


class TestWhereTheDogWaits:
    @pytest.mark.parametrize("floor", (1, 2, 3))
    def test_stair_stop_is_outside_the_stairwell(self, router, floor):
        xy = router.stair_stop(floor)
        assert router.hazard_at(xy) is None
        assert router.clear_of_zones(xy, 0.5)

    def test_stairs_leg_ends_at_the_stair_stop(self, router):
        f, xy, target = trip(router, "room 101", 1, "room 201", 2)
        t = router.choose_transit(f, xy, target)
        legs = router.plan(f, xy, target, transit=t)
        assert len(legs) == 2
        (f1, to_stairs), (f2, onward) = legs
        assert (f1, f2) == (1, 2)
        assert tuple(to_stairs.polyline[-1]) == pytest.approx(t.stop, abs=0.15)
        assert to_stairs.checkpoints[-1].announcements == ["Take the stairs to floor 2"]


class TestNoRouteCrossesAStairwell:
    """The fallback is a deliberate transfer; walking routes still never touch a stairwell."""

    @pytest.mark.parametrize("dest, floor", [("room 101", 1), ("room 106", 1), ("cafeteria", 1)])
    def test_walking_routes_stay_out(self, router, dest, floor):
        f, xy, target = trip(router, "main entrance", 1, dest, floor)
        for leg_floor, route in router.plan(f, xy, target):
            for p in route.polyline:
                assert router.hazard_at(tuple(p)) is None, f"{dest}: route enters a stop zone at {p}"
