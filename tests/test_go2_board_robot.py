"""Go2BoardRobot against go2_board's own end of the link, over localhost UDP."""
import math
import time

import pytest

link = pytest.importorskip("go2_board.link", reason="go2_board not installed")

from cyberdog.sim.robot.go2_board_robot import BoardNotReady, Go2BoardRobot  # noqa: E402
from cyberdog.sim.robot.robot_interface import RobotInterface  # noqa: E402

BRAIN_PORT, BOARD_PORT = 19881, 19880


@pytest.fixture
def pair():
    board = link.BoardLink(port=BOARD_PORT, brain_port=BRAIN_PORT)
    robot = Go2BoardRobot(origin=(10.0, 5.0, math.pi / 2), port=BRAIN_PORT, board_port=BOARD_PORT)
    yield robot, board
    robot.close()
    board.close()


def settle():
    time.sleep(0.03)


def test_implements_the_brains_interface():
    for name in ("get_image", "get_pose", "camera_pose", "get_scan", "now",
                 "set_velocity", "step"):
        assert getattr(Go2BoardRobot, name) is not getattr(RobotInterface, name), name


def test_no_pose_before_the_board_speaks(pair):
    robot, _ = pair
    with pytest.raises(BoardNotReady):
        robot.get_pose()


def test_pose_is_the_boards_odometry_in_the_map_frame(pair):
    robot, board = pair
    board.send_state(x=1.0, y=0.0, z=0.3, yaw=0.0, mode="STAND", tripped=None, walker=False)
    settle()
    x, y, yaw = robot.get_pose()
    # 1 m forward from an origin facing +y is 1 m up the map's y axis.
    assert (x, y, yaw) == pytest.approx((10.0, 6.0, math.pi / 2))
    cx, cy, _ = robot.camera_pose()
    assert (cx, cy) == pytest.approx((10.0, 6.0 + robot.CAM_FORWARD))


def test_a_stale_pose_is_no_pose():
    board = link.BoardLink(port=BOARD_PORT, brain_port=BRAIN_PORT)
    t = [0.0]
    robot = Go2BoardRobot(port=BRAIN_PORT, board_port=BOARD_PORT, clock=lambda: t[0])
    try:
        board.send_state(x=0.0, y=0.0, z=0.3, yaw=0.0, mode="STAND", tripped=None, walker=False)
        settle()
        robot.get_pose()
        t[0] += 0.5
        with pytest.raises(BoardNotReady):
            robot.get_pose()
    finally:
        robot.close()
        board.close()


def test_velocity_is_clipped_and_sent_as_walk_on_step(pair):
    robot, board = pair
    robot.set_velocity(5.0, -5.0, 9.0)
    robot.step()
    settle()
    c = board.poll()
    assert (c.vx, c.vy, c.w, c.mode) == (robot.MAX_V, -robot.MAX_V, robot.MAX_W, "walk")


def test_stand_waits_for_the_board_to_say_so(pair):
    robot, board = pair
    sent = []

    def answer(dt):
        c = board.poll()
        sent.append(c.mode)
        board.send_state(x=0.0, y=0.0, z=0.3, yaw=0.0,
                         mode="STAND" if len(sent) > 3 else "STAND_UP",
                         tripped=None, walker=False)
        time.sleep(0.01)

    robot.sleep = answer
    s = robot.stand(timeout=2.0)
    assert s["mode"] == "STAND" and {m for m in sent if m} == {"stand"}


def test_stand_gives_up_with_what_the_board_said(pair):
    robot, board = pair
    board.send_state(x=0.0, y=0.0, z=0.1, yaw=0.0, mode="LIMP", tripped="tilted 80 deg",
                     walker=False)
    with pytest.raises(BoardNotReady, match="LIMP"):
        robot.stand(timeout=0.2)


def test_sensors_are_not_wired_yet(pair):
    robot, _ = pair
    with pytest.raises(NotImplementedError):
        robot.get_scan()
    with pytest.raises(NotImplementedError):
        robot.get_image()
