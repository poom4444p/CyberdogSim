"""Run.spare_handler's hold: a turn that would put the person on the handle into
a wall waits HOLD_TURN_S, then goes -- and keeps going, not one tick in three s."""
import types

from cyberdog.sim import run_building as rb

DT = 0.05


def stub(pinned=True):
    """A run whose every roll-forward puts the person into the wall (pinned),
    or leaves them plenty of room."""
    run = types.SimpleNamespace(robot=types.SimpleNamespace(CONTROL_DT=DT), turn_held=0, obs={"eased": 0})
    run.room = 0.05 if pinned else 0.6
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.10 if pinned else 0.6
    run.swing = lambda live, pose, cmd, people=(), dt=0.1, need=rb.ROBOT_R: (run.room, True, run.room, 0.5)
    run.spare_handler = rb.Run.spare_handler.__get__(run)
    return run


def ticks(run, n, cmd=(0.0, 0.0, 0.8)):
    return [run.spare_handler(None, (0.0, 0.0, 0.0), cmd, limit=0.5) for _ in range(n)]


HOLD = int(round(rb.HOLD_TURN_S / DT))


def test_a_turn_into_the_wall_is_held_for_hold_turn_s():
    run = stub()
    assert all(c == (0.0, 0.0, 0.0) for c in ticks(run, HOLD))


def test_after_the_hold_the_turn_goes_on_not_for_one_tick():
    # Out of the floor-3 lift, the user against the car and the wall: the old
    # count reset after the one tick it let through, so the dog turned 0.04
    # rad every 3 s and stood there for 215 s.
    run = stub()
    ticks(run, HOLD)
    after = ticks(run, 40)
    assert all(c[2] != 0.0 for c in after), [c for c in after if c[2] == 0.0][:3]


def test_free_again_the_next_pinned_turn_waits_again():
    run = stub()
    ticks(run, HOLD + 5)
    run.room = 0.6                                   # the turn no longer reaches the wall
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.6
    assert ticks(run, 1) == [(0.0, 0.0, 0.8)] and run.turn_held == 0
    run.room = 0.05
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.10
    assert ticks(run, 1) == [(0.0, 0.0, 0.0)]       # pinned anew: held anew


def test_a_turn_with_room_is_not_held():
    run = stub(pinned=False)
    assert ticks(run, 3) == [(0.0, 0.0, 0.8)] * 3
