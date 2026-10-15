"""The two crowds of scene/pedestrians.py: oblivious (the stress test) and aware.

No MuJoCo: walkers are plain geometry, stepped by hand next to a dog that is
just a pose. The party is the dog and its user HANDLER_BEHIND straight behind.
"""
import math

from cyberdog.sim.scene import obstacles, pedestrians as ped
from cyberdog.sim.scene.pedestrians import Crowd, Walker

DT = 0.05
WEST = math.pi


def lateral(w, dog):
    """How far the walker is off the party's line."""
    x, y, yaw = dog
    return abs(-(w.x - x) * math.sin(yaw) + (w.y - y) * math.cos(yaw))


def test_oblivious_ignores_the_heading():
    # Every campaign before the aware crowd was scored with (x, y) only; handing
    # the oblivious crowd a heading as well must not move anybody differently.
    a, b = Crowd(1, 3, seed=5), Crowd(1, 3, seed=5)
    for k in range(2000):
        dog = (40.0 - 0.02 * k, 10.1, WEST)
        a.step(DT, dog[:2])
        b.step(DT, dog)
    assert a.positions() == b.positions()


def test_aware_without_a_heading_falls_back_to_oblivious():
    a, b = Crowd(1, 3, seed=5), Crowd(1, 3, seed=5, aware=True)
    for k in range(400):
        dog = (40.0 - 0.02 * k, 10.1)
        a.step(DT, dog)
        b.step(DT, dog)
    assert a.positions() == b.positions()


def test_crosser_waits_at_the_side_until_the_party_has_passed():
    # Starting from the south wall, about to cross where the dog is heading.
    w = Walker((20.0, ped.LANE_Y[0]), (20.0, ped.LANE_Y[1]), 1.2, 0.0, "crossing")
    dog = [23.0, 10.1, WEST]
    worst = math.inf
    for _ in range(int(12 / DT)):
        w.step(DT, tuple(dog), aware=True)
        along = (w.x - dog[0]) * math.cos(dog[2]) + (w.y - dog[1]) * math.sin(dog[2])
        if -ped.PARTY_BACK < along < ped.AWARE_R:
            worst = min(worst, lateral(w, dog))
        dog[0] -= 0.5 * DT                     # the party walks west at 0.5 m/s
    assert worst >= ped.PASS_ROOM - 0.05       # never stepped into its path
    assert w.done or w.t > 0.9                 # and crossed once it had gone by


def test_crosser_already_in_the_path_gets_out_of_it():
    w = Walker((20.0, ped.LANE_Y[0]), (20.0, ped.LANE_Y[1]), 1.2, 0.0, "crossing", phase=0.6)
    dog = (23.0, 10.1, WEST)
    y0 = w.y
    for _ in range(int(1.0 / DT)):
        w.step(DT, dog, aware=True)
    assert w.y > y0 + 0.3                      # kept going, did not stop in it


def test_nobody_stands_in_front_of_the_dog():
    # Oblivious: a walker the dog is in front of holds for GIVE_WAY seconds.
    # Aware: they step aside on the first tick.
    w = Walker((20.0, ped.LANE_Y[0]), (20.0, ped.LANE_Y[1]), 1.2, 0.0, "crossing", phase=0.75)
    dog = (20.5, w.y, WEST)
    w.step(DT, dog, aware=True)
    assert w.held == 0.0 and w.a == (w.x, w.y) and w.b != (20.0, ped.LANE_Y[1])


def test_head_on_walker_moves_over_before_passing():
    # Walking east straight down the party's line.
    w = Walker((10.0, 10.1), (30.0, 10.1), 1.2, 0.0, "along")
    dog = [22.0, 10.1, WEST]
    worst = math.inf
    for _ in range(int(8 / DT)):
        w.step(DT, tuple(dog), aware=True)
        along = (w.x - dog[0]) * math.cos(dog[2]) + (w.y - dog[1]) * math.sin(dog[2])
        if -ped.PARTY_BACK < along < 0.5:      # level with the dog or its user
            worst = min(worst, lateral(w, dog))
        dog[0] -= 0.5 * DT
    assert worst >= 0.55                       # PED_NEAR: no contact in passing


def test_follower_does_not_tailgate_when_there_is_no_room_to_pass():
    # The party in the middle of the corridor: neither side has PASS_ROOM.
    w = Walker((30.0, 9.5), (5.0, 9.5), 1.4, 0.0, "along")
    dog = [26.0, 9.5, WEST]
    closest = math.inf
    for _ in range(int(10 / DT)):
        w.step(DT, tuple(dog), aware=True)
        closest = min(closest, w.x - dog[0])
        dog[0] -= 0.5 * DT
    assert closest >= ped.PARTY_BACK + ped.FOLLOW_GAP - 0.1


def test_follower_overtakes_with_room_when_there_is_some():
    # Party hugging the north wall: the south side has room.
    w = Walker((30.0, 10.1), (5.0, 10.1), 1.4, 0.0, "along")
    dog = [26.0, 10.1, WEST]
    worst = math.inf
    for _ in range(int(14 / DT)):
        w.step(DT, tuple(dog), aware=True)
        along = (w.x - dog[0]) * math.cos(dog[2]) + (w.y - dog[1]) * math.sin(dog[2])
        if -ped.PARTY_BACK < along < 0.5:
            worst = min(worst, lateral(w, dog))
        dog[0] -= 0.5 * DT
    assert w.x < dog[0]                        # got past
    assert worst >= ped.PASS_ROOM - 0.1        # with room


def test_head_on_walker_on_the_wall_side_passes_on_the_open_side():
    # The party hugs the north wall; the walker comes the other way between
    # it and that wall, where there is not PASS_ROOM. It has to go across.
    w = Walker((10.0, 10.5), (30.0, 10.5), 1.2, 0.0, "along")
    dog = [22.0, 10.1, WEST]
    worst = math.inf
    for _ in range(int(10 / DT)):
        w.step(DT, tuple(dog), aware=True)
        along = (w.x - dog[0]) * math.cos(dog[2]) + (w.y - dog[1]) * math.sin(dog[2])
        if -ped.PARTY_BACK < along < 0.5:
            worst = min(worst, lateral(w, dog))
        dog[0] -= 0.5 * DT
    assert worst >= ped.PASS_ROOM - 0.1


def _overlaps(w, boxes):
    return any(x0 - ped.RADIUS < w.x < x1 + ped.RADIUS and y0 - ped.RADIUS < w.y < y1 + ped.RADIUS
               for x0, x1, y0, y1, _h, _n in boxes)


def test_walkers_go_round_crates_not_through():
    # Room 101 seed 6, aware: a walker drifted onto a line through the floor-1
    # trolley and walked through it. Every line through a crate, either way,
    # either axis: never inside it, and never a jump sideways.
    boxes = obstacles.boxes(1)
    legs = [((30.0, y), (17.0, y)) for y in (9.1, 9.4, 9.7)]
    legs += [((17.0, 9.4), (30.0, 9.4))]
    legs += [((x, ped.LANE_Y[0]), (x, ped.LANE_Y[1])) for x in (23.5, 24.0, 24.5)]
    for a, b in legs:
        w = Walker(a, b, 1.45, 0.0, "along", boxes=boxes)
        last = (w.x, w.y)
        for _ in range(400):
            w.step(0.05)
            if w.done:
                break
            assert not _overlaps(w, boxes), (a, b, w.x, w.y)
            assert math.dist(last, (w.x, w.y)) < 0.15, (a, b, last, (w.x, w.y))
            last = (w.x, w.y)


def test_floor2_south_lane_clears_the_crate():
    # The south lane (y 8.60) used to clip the floor-2 crate (y from 8.75).
    boxes = obstacles.boxes(2)
    w = Walker((15.0, ped.SIDE_LANES[0]), (40.0, ped.SIDE_LANES[0]), 1.2, 0.0, "along", boxes=boxes)
    for _ in range(600):
        w.step(0.05)
        if w.done:
            break
        assert not _overlaps(w, boxes), (w.x, w.y)
