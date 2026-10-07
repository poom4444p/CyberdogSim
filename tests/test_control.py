"""Tests for sim/control.py: when a waypoint counts as reached."""
from cyberdog.sim.control import advance

# Floor 3 to the chemistry lab: west along the corridor, with a 10 cm jog
# north and a repeated point at (43.26, 10.0-10.16) on the way.
ROUTE = [(45.0, 9.5), (43.26, 10.0), (43.26, 10.1), (43.26, 10.1),
         (43.25, 10.16), (40.77, 10.15), (40.48, 10.12)]


class TestAdvance:
    def test_not_yet_at_the_first_waypoint(self):
        assert advance(1, 44.5, 9.6, ROUTE) == 1

    def test_reached_inside_arrive_r(self):
        # On the jog: its first three points are within ARRIVE_R, the fourth
        # (43.25, 10.16) is not, and the dog is not yet past it going west.
        assert advance(1, 43.3, 10.0, ROUTE) == 4

    def test_walked_past_a_jog_off_its_line(self):
        # 1.8 m west of the jog, never north of it: the jog is behind the dog.
        assert advance(2, 41.5, 9.97, ROUTE) == 5

    def test_a_real_waypoint_ahead_stays_ahead(self):
        assert advance(5, 41.5, 9.97, ROUTE) == 5

    def test_overshoot_on_a_long_segment_still_advances(self):
        # Past (40.77, 10.15) going west, 0.3 m off the line.
        assert advance(5, 40.6, 9.85, ROUTE) == 6

    def test_the_last_waypoint_is_never_skipped(self):
        assert advance(len(ROUTE) - 1, 0.0, 0.0, ROUTE) == len(ROUTE) - 1
