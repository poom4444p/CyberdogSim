"""The brain's side of the Go2 board (go2_board, on the J4012).

Same RobotInterface as the twin, so run_building drives it unchanged:
set_velocity goes to the board as vx, vy, w over the localhost link, and
get_pose comes back from the board's state messages. The board owns the
legs, safety and the walking policy; nothing here touches a motor.

Needs go2_board installed (pip install -e ~/Work/unitree/go2_board). The
import is inside __init__, so the twin runs without it.

Not yet: get_scan and get_image. In the sim they come from the physics twin
of the building (B6); on the dog, from the Mid-360 and the camera.
"""
import math
import time

from cyberdog.sim.robot.robot_interface import RobotInterface

STATE_STALE_S = 0.2       # the board reports at 50 Hz: 10 missed reports


class BoardNotReady(RuntimeError):
    pass


class Go2BoardRobot(RobotInterface):
    CONTROL_DT  = 0.05         # 20 Hz, the twin's rate: the brain is tuned at it
    MAX_V       = 0.8          # m/s   -- the twin's limits. The Go2 policy is
    MAX_W       = 1.5          # rad/s    trained to +-1 rad/s: check before WALK
    CAM_FORWARD = 0.34         # m, lens ahead of body centre: measure on the dog

    def __init__(self, origin=(0.0, 0.0, 0.0), port=None, board_port=None,
                 clock=time.monotonic, sleep=time.sleep):
        """`origin` is where the board's odometry frame starts, in the map
        frame: (x, y, yaw) of the dog when the board came up. FAST-LIO starts
        wherever it was switched on; the map is the building's."""
        from go2_board.link import BOARD_PORT, BRAIN_PORT, BrainLink
        self.link = BrainLink(port=port or BRAIN_PORT, board_port=board_port or BOARD_PORT)
        self.origin = origin
        self.clock, self.sleep = clock, sleep
        self._t0 = clock()
        self._next = self._t0 + self.CONTROL_DT
        self._cmd = (0.0, 0.0, 0.0)
        self._state, self._got = None, None

    # -- from the board ------------------------------------------------------

    def board_state(self):
        """The newest state message, or None. Never older than STATE_STALE_S."""
        s = self.link.poll()
        if s is not None and s is not self._state:
            self._state, self._got = s, self.clock()
        if self._got is None or self.clock() - self._got > STATE_STALE_S:
            return None
        return self._state

    def get_pose(self):
        s = self.board_state()
        if s is None or s.get("x") is None:
            raise BoardNotReady("no fresh pose from the board")
        x0, y0, yaw0 = self.origin
        c, k = math.cos(yaw0), math.sin(yaw0)
        x, y = s["x"], s["y"]
        return (x0 + c * x - k * y, y0 + k * x + c * y,
                math.atan2(math.sin(yaw0 + s["yaw"]), math.cos(yaw0 + s["yaw"])))

    def camera_pose(self):
        x, y, yaw = self.get_pose()
        return (x + self.CAM_FORWARD * math.cos(yaw),
                y + self.CAM_FORWARD * math.sin(yaw), yaw)

    def get_scan(self):
        raise NotImplementedError("B6: the LiDAR from the physics twin, or the Mid-360")

    def get_image(self):
        raise NotImplementedError("B6: the camera from the physics twin, or the dog's")

    def now(self):
        return self.clock() - self._t0

    # -- to the board --------------------------------------------------------

    def set_velocity(self, vx, vy=0.0, w=0.0):
        self._cmd = (max(-self.MAX_V, min(self.MAX_V, float(vx))),
                     max(-self.MAX_V, min(self.MAX_V, float(vy))),
                     max(-self.MAX_W, min(self.MAX_W, float(w))))

    def step(self):
        """Send this tick's command, then wait for the next 20 Hz tick."""
        self.link.send(*self._cmd, mode="walk")
        wait = self._next - self.clock()
        if wait > 0:
            self.sleep(wait)
            self._next += self.CONTROL_DT
        else:
            self._next = self.clock() + self.CONTROL_DT

    def request(self, mode, until=None, timeout=5.0):
        """Ask the board for a mode ("stand", "lie", "limp") at 20 Hz, with
        zero velocity, until its state says `until` (default: no wait)."""
        end = self.clock() + timeout
        while True:
            self.link.send(0.0, 0.0, 0.0, mode=mode)
            s = self.board_state()
            if until is None or (s is not None and s["mode"] == until):
                return s
            if self.clock() > end:
                got = None if self._state is None else self._state["mode"]
                raise BoardNotReady(f"asked {mode!r}, board still {got!r} after {timeout:.0f} s")
            self.sleep(self.CONTROL_DT)

    def stand(self, timeout=5.0):
        return self.request("stand", until="STAND", timeout=timeout)

    def lie(self, timeout=5.0):
        return self.request("lie", until="LIMP", timeout=timeout)

    def close(self):
        self.link.close()
