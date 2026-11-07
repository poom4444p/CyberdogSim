"""Run.step_aside on go2_board: the board's LiDAR guard has the last word on
the room sideways, as in respect_board."""
import math
import types

from cyberdog.sim import run_building as rb

OPEN = 5.0


class Live:
    """Nothing near the dog in the brain's own view."""
    def static_at(self, x, y):
        return OPEN

    detected_at = wall_at = remembered_at = static_at


class Board:
    REAL_TIME = True

    def __init__(self, left=1.0, right=1.0):
        self.free = {"front": OPEN, "back": math.inf, "left": left, "right": right}

    def board_clearance(self):
        return self.free


class Twin:
    """MujocoRobot: no guard of its own."""


def oncoming():
    """Walking at the dog (at the origin, facing +x), a little to its right:
    stepping left gets out of the way, stepping right walks into it."""
    return types.SimpleNamespace(moving=True, walked=True, speed=1.0,
                                 predict=lambda t: (2.0 - 1.0 * t, -0.4))


def run_with(robot):
    said = []
    run = types.SimpleNamespace(robot=robot, evading=0.0, evade={"steps": 0, "ticks": 0},
                                say=said.append)
    run.handler_room = rb.Run.handler_room.__get__(run)
    run.step_aside = rb.Run.step_aside.__get__(run)
    return run, said


def test_board_with_room_steps_away_as_before():
    run, said = run_with(Board(left=1.0, right=1.0))
    assert run.step_aside(Live(), [oncoming()], (0.0, 0.0, 0.0)) == rb.SIDE_V
    assert said == ["Someone close. Stepping to your left."]


def test_board_with_no_room_that_way_does_not_step_there():
    # The brain sees the left open; the board sees 0.09 m (the trolley case).
    run, said = run_with(Board(left=0.09, right=1.0))
    assert run.step_aside(Live(), [oncoming()], (0.0, 0.0, 0.0)) == 0.0   # right is worse: stay
    assert said == [] and run.evading == 0.0


def test_room_exactly_at_the_threshold_is_not_enough():
    run, _ = run_with(Board(left=rb.BOARD_SIDE_ROOM, right=1.0))
    assert run.step_aside(Live(), [oncoming()], (0.0, 0.0, 0.0)) == 0.0


def test_blind_board_gets_no_step():
    board = Board()
    board.free = None
    run, said = run_with(board)
    assert run.step_aside(Live(), [oncoming()], (0.0, 0.0, 0.0)) == 0.0
    assert said == []


def test_the_twin_has_no_guard_and_steps_as_before():
    run, _ = run_with(Twin())
    assert run.step_aside(Live(), [oncoming()], (0.0, 0.0, 0.0)) == rb.SIDE_V
