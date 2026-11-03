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


# -- BoardTwinRobot: the twin as the board's shadow ----------------------------

from cyberdog import paths  # noqa: E402


@pytest.fixture
def twin():
    if not paths.BUILDING_SCENE.exists():
        pytest.skip("no building scene: python -m cyberdog.sim.scene.build_scene --building")
    from cyberdog.sim.robot.go2_board_robot import BoardTwinRobot
    board = link.BoardLink(port=BOARD_PORT, brain_port=BRAIN_PORT)
    brain = Go2BoardRobot(origin=(21.0, 9.5, 0.0), port=BRAIN_PORT, board_port=BOARD_PORT)
    board.send_state(x=1.0, y=0.5, z=0.35, yaw=math.pi / 2, mode="WALK", tripped=None,
                     walker=True)
    settle()
    robot = BoardTwinRobot(str(paths.BUILDING_SCENE), board=brain)
    yield robot, board
    robot.close()
    board.close()


def test_twin_starts_where_the_board_is(twin):
    robot, _ = twin
    x, y, yaw = robot.get_pose()
    assert (x, y, yaw) == pytest.approx((22.0, 10.0, math.pi / 2), abs=1e-6)
    cx, cy, _ = robot.camera_pose()
    assert (cx, cy) == pytest.approx((22.0, 10.0 + robot.CAM_FORWARD), abs=1e-6)


def test_twin_sends_the_command_and_follows_the_board(twin):
    robot, board = twin
    robot.set_velocity(0.5, 0.0, 3.0)
    board.send_state(x=2.0, y=0.5, z=0.35, yaw=0.0, mode="WALK", tripped=None, walker=True)
    settle()
    robot.step()
    settle()
    c = board.poll()
    assert (c.vx, c.w, c.mode) == (0.5, robot.MAX_W, "walk") and robot.MAX_W == 1.0
    assert robot.get_pose() == pytest.approx((23.0, 10.0, 0.0), abs=1e-6)


def test_twin_walks_to_a_placed_pose_but_cannot_be_carried(twin):
    robot, board = twin
    x, y, yaw = robot.get_pose()                       # (22, 10), facing +y
    board.send_state(x=1.0, y=0.5, z=0.35, yaw=math.pi / 2, mode="WALK", tripped=None,
                     walker=True)
    settle()
    robot.place((x, y + 2.0), yaw)                     # 2 m straight ahead
    settle()
    c = board.poll()
    assert c.vx == pytest.approx(robot.TRACK_V) and abs(c.vy) < 1e-9 and abs(c.w) < 1e-9
    robot.place((x, y), yaw)                           # already there: stand still
    settle()
    c = board.poll()
    assert (c.vx, c.vy, c.w) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    with pytest.raises(NotImplementedError, match="lift"):
        robot.place((x, y), yaw, z=robot.z + 1.0)      # a scripted climb


class FakeLift:
    def __init__(self):
        self.called = []

    def call(self, floor):
        self.called.append(floor)

    def close(self):
        pass


def test_ride_stands_waits_for_its_own_height_then_walks(twin):
    robot, board = twin
    robot.lift = FakeLift()
    seen, t = [], [0.0]
    heights = iter([0.35] * 3 + [0.35 + 0.1 * k for k in range(1, 23)] + [2.55] * 40)

    def answer(dt):
        t[0] += dt
        c = board.poll()
        seen.append(c.mode)
        mode = "WALK" if seen.count("walk") > 0 else "STAND"
        board.send_state(x=1.0, y=0.5, z=next(heights, 2.55), yaw=0.0, mode=mode,
                         tripped=None, walker=True)
        time.sleep(0.003)

    robot.board.clock, robot.board.sleep = (lambda: t[0]), answer
    ticks = []
    robot.ride_lift(2, 2.2, ticks.append)
    assert robot.lift.called == [2]
    assert seen.index("walk") > len(seen) - 5 and set(seen[:-3]) <= {"stand", None}
    assert ticks[-1] == pytest.approx(2.2, abs=0.01) and robot.z == 2.2

def test_twin_start_asks_afresh_after_a_trip(twin):
    robot, board = twin
    seen, modes, t = [], iter(["LIMP"] * 8 + ["STAND"] * 2), [0.0]

    def answer(dt):
        t[0] += dt
        c = board.poll()
        seen.append(c.mode)
        board.send_state(x=1.0, y=0.5, z=0.35, yaw=0.0, mode=next(modes, "WALK"),
                         tripped=None, walker=True)
        time.sleep(0.005)

    robot.board.clock, robot.board.sleep = (lambda: t[0]), answer
    robot.start(timeout=0.12)
    seen = [m for m in seen if m]                  # None: before the first datagram landed
    asked = [m for k, m in enumerate(seen) if k == 0 or m != seen[k - 1]]
    assert asked == ["stand", "limp", "stand", "walk"]
    assert seen.count("limp") >= robot.LIMP_HOLD - 1      # held, not one datagram


def test_twin_stops_the_run_when_the_board_stops_walking(twin):
    from cyberdog.sim.robot.go2_board_robot import BoardStopped
    robot, board = twin
    board.send_state(x=1.0, y=0.5, z=0.12, yaw=0.0, mode="LIMP",
                     tripped="stale state (55 ms old)", walker=True)
    settle()
    with pytest.raises(BoardStopped, match="LIMP, not walking: stale state"):
        robot.step()


def test_board_host_and_listen_reach_the_link():
    robot = Go2BoardRobot(port=19893, board_port=19892, board_host="127.0.0.1", listen="127.0.0.1")
    try:
        assert robot.link.board == ("127.0.0.1", 19892)
        assert robot.link.sock.getsockname() == ("127.0.0.1", 19893)
    finally:
        robot.close()


def test_scan_is_the_boards_cloud_in_the_map_frame(twin):
    import socket
    import numpy as np
    from go2_board.lidar import BRAIN_CLOUD_PORT, cloud_datagrams
    robot, board = twin                                # at map (22, 10), facing +y
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        pts = np.array([[2.0, 0.0, 0.5], [0.0, -1.0, 0.0]], np.float32)   # ahead, and to the right
        for d in cloud_datagrams(1, pts, 0, 0.0):
            tx.sendto(d, ("127.0.0.1", BRAIN_CLOUD_PORT))
        for d in cloud_datagrams(2, pts, 0, 0.0):      # a frame closes when the next begins
            tx.sendto(d, ("127.0.0.1", BRAIN_CLOUD_PORT))
        settle()
        scan = robot.get_scan()
        assert scan.shape == (2, 3)
        assert scan[0] == pytest.approx((22.0, 12.0, robot.z + 0.5), abs=1e-5)
        assert scan[1] == pytest.approx((23.0, 10.0, robot.z), abs=1e-5)
        robot.board.clock = lambda: robot.board.sleep and 1e9          # long after: stale
        assert robot.get_scan().shape == (0, 3)
    finally:
        tx.close()
