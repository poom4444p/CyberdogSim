"""Run.respect_board: the brain works with the board's LiDAR guard, never
backs into the person on the handle, and asks for help when it cannot pass."""
import math
import types

from cyberdog.sim import run_building as rb


class Robot:
    def __init__(self):
        self.t, self.free = 0.0, None

    def board_clearance(self):
        return self.free

    def now(self):
        return self.t


def stub(free):
    robot = Robot()
    robot.free = free
    said = []
    run = types.SimpleNamespace(robot=robot, board_turning=None, board_since=0.0, board_push=None,
                                board_gave_up=False, gave_up=None, obs={}, say=said.append)
    run.respect_board = rb.Run.respect_board.__get__(run)
    return run, said


def open_(front=5.0, left=1.0, right=1.0):
    return {"front": front, "back": math.inf, "left": left, "right": right}


def test_never_backwards_with_a_person_on_the_handle():
    run, _ = stub(open_())
    assert run.respect_board((-0.3, 0.1, 0.2), (0, 0, 0)) == (0.0, 0.1, 0.2)


def test_open_ahead_passes_the_plan_through():
    run, said = stub(open_())
    assert run.respect_board((0.5, 0.0, 0.1), (0, 0, 0)) == (0.5, 0.0, 0.1) and not said


def test_pushed_against_the_guard_turns_to_the_roomier_side_then_carries_on():
    run, said = stub(open_(front=0.1, left=0.2, right=0.9))
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0)) == (0.0, 0.0, -rb.BOARD_TURN_W)
    assert said and "right" in said[0]
    run.robot.t = 1.0
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0))[2] == -rb.BOARD_TURN_W   # still closed
    run.robot.free = open_(front=rb.BOARD_CLEAR_M + 0.1)
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0)) == (0.4, 0.0, 0.0)
    assert run.board_turning is None and len(said) == 1


def test_asks_for_help_when_turning_does_not_open_the_way():
    run, said = stub(open_(front=0.05))
    run.respect_board((0.4, 0.0, 0.0), (3.0, 4.0, 0.0))
    run.robot.t = rb.BOARD_GIVE_UP_S + 0.1
    assert run.respect_board((0.4, 0.0, 0.0), (3.0, 4.0, 0.0)) == (0.0, 0.0, 0.0)
    assert run.board_gave_up and run.gave_up[0] == "go2_board" and "help" in said[-1]


def test_blind_board_leaves_the_plan_alone_but_still_never_backwards():
    run, _ = stub(None)
    assert run.respect_board((-0.2, 0.0, 0.3), (0, 0, 0)) == (0.0, 0.0, 0.3)


def test_crawled_to_a_stop_by_the_guard_counts_as_stuck():
    # 0.5 m free: past BOARD_NEAR_M, but the guard gives too little to step at
    run, said = stub(open_(front=0.5, left=1.0, right=0.3))
    for k in range(30):
        run.robot.t = 0.1 * k
        out = run.respect_board((0.3, 0.0, 0.0), (44.6, 8.7, 0.0))      # not moving
    assert out == (0.0, 0.0, rb.BOARD_TURN_W) and "left" in said[0]


def test_walking_on_is_not_stuck():
    run, said = stub(open_(front=0.5))
    for k in range(40):
        run.robot.t = 0.1 * k
        out = run.respect_board((0.3, 0.0, 0.0), (44.0 + 0.03 * k, 8.7, 0.0))
    assert out == (0.3, 0.0, 0.0) and not said
