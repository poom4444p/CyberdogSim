"""Run.glide: the scripted few metres in and out of the lift car end where they
were meant to, on a robot that walks after its target (go2_board) as on one
that is put on it (the twin)."""
import math
import types

from cyberdog.sim import run_building as rb

DT = 0.05


class Robot:
    """place() moves `gain` of the way to the target each tick: 1.0 is the
    twin, set on it; less is a board walking after it at its own limits."""
    CONTROL_DT = DT

    def __init__(self, pose, gain):
        self.pose, self.gain, self.calls = list(pose), gain, 0

    def get_pose(self):
        return tuple(self.pose)

    def place(self, xy, yaw, z=None):
        self.calls += 1
        x, y, th = self.pose
        g = self.gain
        self.pose = [x + g * (xy[0] - x), y + g * (xy[1] - y), th + g * rb.wrap(yaw - th)]


def run_with(robot):
    run = types.SimpleNamespace(robot=robot, handle_wait=lambda: None, step_crowd=lambda: None,
                                frame=lambda: None)
    run.glide = rb.Run.glide.__get__(run)
    return run


def test_a_board_that_lags_still_ends_at_the_waiting_spot():
    # Out of the floor-3 car to the waiting spot, turning round: the line ran
    # out with the board 1.2 m and 1.3 rad short of it.
    robot = Robot((46.5, 9.5, 0.43), gain=0.10)
    run_with(robot).glide((45.0, 9.5), math.pi, 4.4)
    x, y, th = robot.get_pose()
    assert math.dist((x, y), (45.0, 9.5)) <= rb.GLIDE_TOL_M
    assert abs(rb.wrap(math.pi - th)) <= rb.GLIDE_TOL_YAW


def test_the_twin_takes_no_extra_ticks():
    robot = Robot((46.5, 9.5, 0.43), gain=1.0)
    run_with(robot).glide((45.0, 9.5), math.pi, 4.4)
    line = max(int(1.5 / (rb.STEP_V * DT)), 1)
    assert robot.calls == line


def test_a_board_that_cannot_get_there_gives_up_after_glide_settle_s():
    robot = Robot((46.5, 9.5, 0.43), gain=0.0)                      # never moves
    run_with(robot).glide((45.0, 9.5), math.pi, 4.4)
    line = max(int(1.5 / (rb.STEP_V * DT)), 1)
    assert robot.calls == line + int(rb.GLIDE_SETTLE_S / DT)
