"""run_building.describe_person: what the person on the handle is told about
somebody the dog is waiting for. They hear it and cannot see it."""
import math

from cyberdog.sim.run_building import describe_person

WEST = math.pi
DOG = (24.7, 10.1)     # room 101 seed 6, beside the floor-1 trolley, heading west


def test_overtaker_on_the_left_is_not_a_crossing_from_the_right():
    # 35 s of the trace: 2 m behind, 1 m to the dog's left (south), walking
    # west past it. Was "Someone crossing from the right".
    assert describe_person(DOG, WEST, (26.7, 9.1), (-1.24, 0.09)) == "Someone overtaking, on your left"


def test_oncoming_in_the_far_lane():
    # 44 s: ahead, 1.3 m to the left, walking east. Was "crossing from the right".
    assert (describe_person((23.6, 10.1), WEST, (19.3, 8.8), (1.02, 0.07))
            == "Someone coming towards us, on your left")


def test_crossing_names_the_side_they_come_from():
    # Heading east, so left is north. Walking south: they come from the left.
    assert describe_person((10.0, 9.5), 0.0, (12.0, 10.4), (0.0, -1.1)) == "Someone crossing from the left"
    assert describe_person((10.0, 9.5), 0.0, (12.0, 8.6), (0.0, 1.1)) == "Someone crossing from the right"


def test_head_on_on_the_line_has_no_side():
    assert describe_person((10.0, 9.5), 0.0, (13.0, 9.6), (-1.2, 0.0)) == "Someone coming towards us"


def test_standing_still():
    assert describe_person((10.0, 9.5), 0.0, (11.0, 9.5), (0.0, 0.05)) == "Someone ahead"
    assert describe_person((10.0, 9.5), 0.0, (11.0, 8.5), (0.05, 0.0)) == "Someone on your right"


def test_slow_sideways_drift_is_not_a_crossing():
    # 0.1 m/s across while walking along: that sign is noise.
    assert describe_person(DOG, WEST, (21.0, 9.0), (1.0, 0.15)).startswith("Someone coming towards us")
