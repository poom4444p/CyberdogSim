"""CE-RRT* on synthetic scenes: no map, no sim, a corridor drawn in `ok`."""
import math
import time

import pytest

from cyberdog.planning import ce_rrt
from cyberdog.planning.ce_rrt import Person, plan

HALF_W = 1.25           # a 2.5 m corridor along x, centred on y = 0
ROBOT = 0.16


def corridor(boxes=()):
    """`ok` for a corridor with axis-aligned boxes (x0, x1, y0, y1) in it."""
    def ok(x, y, yaw):
        if abs(y) > HALF_W - ROBOT:
            return False
        for x0, x1, y0, y1 in boxes:
            if x0 - ROBOT <= x <= x1 + ROBOT and y0 - ROBOT <= y <= y1 + ROBOT:
                return False
        return True
    return ok


def walk(path, v=ce_rrt.V):
    """(x, y, t, yaw) every 5 cm along `path` walked at `v`."""
    out, s = [], 0.0
    for a, b in zip(path, path[1:]):
        d = math.dist(a, b)
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        n = max(int(d / 0.05), 1)
        for k in range(n):
            f = k / n
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                        (s + d * f) / v, yaw))
        s += d
    return out


def test_empty_corridor_is_a_straight_line():
    p = plan((0.0, 0.0, 0.0), (4.0, 0.0), corridor())
    assert p is not None
    assert math.dist(p.path[-1], (4.0, 0.0)) <= ce_rrt.GOAL_TOL
    assert len(p.path) == 2              # shortcut to one edge


def test_goes_round_a_crate_and_keeps_clear():
    crate = (1.5, 2.5, -0.6, 0.6)
    ok = corridor([crate])
    p = plan((0.0, 0.0, 0.0), (4.0, 0.0), ok)
    assert p is not None
    assert all(ok(x, y, yaw) for x, y, _, yaw in walk(p.path))


def test_turns_are_curves_not_pivots():
    p = plan((0.0, 0.0, 0.0), (4.0, 0.0), corridor([(1.5, 2.5, -0.6, 0.6)]))
    yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(p.path, p.path[1:])]
    assert abs(yaws[0]) <= ce_rrt.FIRST_TURN + 1e-9
    for a, b in zip(yaws, yaws[1:]):
        assert abs(ce_rrt._wrap(b - a)) <= ce_rrt.MAX_TURN + 1e-9


def test_passes_someone_walking_towards_it_with_the_chance_margin():
    # Head-on, a metre off the dog's line, at a walking pace.
    them = Person(4.0, 0.3, -1.0, 0.0)
    p = plan((0.0, 0.0, 0.0), (4.5, 0.0), corridor(), people=[them])
    assert p is not None
    for x, y, t, _ in walk(p.path):
        if t > ce_rrt.PREDICT_T:
            break
        px, py = them.x + them.vx * t, them.y + them.vy * t
        assert math.hypot(x - px, y - py) >= ce_rrt.need(t) - ce_rrt.REL_CHECK
    # Round them, not through them: they are on the left, it went right.
    assert p.side == -1.0


def test_no_plan_when_the_corridor_is_shut():
    # Somebody standing square in a corridor too narrow to pass them in.
    narrow = lambda x, y, yaw: abs(y) <= 0.45
    p = plan((0.0, 0.0, 0.0), (4.0, 0.0), narrow, people=[Person(2.0, 0.0, 0.0, 0.0)])
    assert p is None


def test_no_plan_from_somewhere_the_rules_already_reject():
    assert plan((0.0, 1.2, 0.0), (4.0, 0.0), corridor()) is None


def test_same_seed_same_plan():
    args = ((0.0, 0.0, 0.0), (4.0, 0.0), corridor([(1.5, 2.5, -0.6, 0.6)]))
    assert plan(*args, seed=7).path == plan(*args, seed=7).path


def test_warm_start_keeps_the_side():
    them = Person(4.0, 0.3, -1.0, 0.0)
    first = plan((0.0, 0.0, 0.0), (4.5, 0.0), corridor(), people=[them])
    again = plan((0.0, 0.0, 0.0), (4.5, 0.0), corridor(), people=[them],
                 warm=first.path, seed=99)
    assert again is not None and again.side == first.side


def test_fast_enough_for_the_control_loop():
    # Spec L6: >= 10 Hz, so a plan must fit inside the REPLAN_S it is
    # asked at. Measured at 15-40 ms on an M-series Mac.
    them = [Person(4.0, 0.3, -1.0, 0.0), Person(6.0, -0.5, -0.8, 0.1)]
    ok = corridor([(2.5, 3.0, 0.6, 1.25)])
    t0 = time.perf_counter()
    for s in range(5):
        plan((0.0, 0.0, 0.0), (4.5, 0.0), ok, people=them, seed=s)
    per = (time.perf_counter() - t0) / 5
    assert per < ce_rrt.REPLAN_S, f"{per * 1000:.0f} ms per plan"


def test_lets_someone_overtake():
    # From behind, faster than the dog, on its left: it keeps out of their way.
    them = Person(-3.0, 0.5, 1.2, 0.0)
    p = plan((0.0, 0.0, 0.0), (4.5, 0.0), corridor(), people=[them])
    assert p is not None
    for x, y, t, _ in walk(p.path):
        if t > ce_rrt.PREDICT_T:
            break
        px, py = them.x + them.vx * t, them.y + them.vy * t
        assert math.hypot(x - px, y - py) >= ce_rrt.need(t) - ce_rrt.REL_CHECK


def test_goal_stays_short_of_a_corner():
    # run_building's goal for a going-round: along the straight, never round
    # the corner -- the plan that cut the cafeteria corner left the dog in
    # the door frame until the run timed out.
    from cyberdog.sim.run_building import PLAN_D, PLAN_SHORT, Run
    goal = Run.straight_goal
    assert goal((0, 0), [(10, 0)]) == (PLAN_D, 0.0)                  # open straight
    assert goal((0, 0), [(3, 0), (3, 5)]) == (3 - PLAN_SHORT, 0.0)   # a corner at 3 m
    assert goal((0, 0), [(2, 0), (2, 5)]) is None                    # too close to plan
    assert goal((0, 0), [(1, 0)]) is None                            # the route ends
    g = goal((0, 0), [(1, 0), (1.5, 0.1), (6, 0)])                   # a jog is no corner
    assert g is not None and abs(g[0] - PLAN_D) < 0.05
