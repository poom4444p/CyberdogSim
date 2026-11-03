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
import json
import math
import socket
import time

import numpy as np

from cyberdog.sim.robot.robot_interface import RobotInterface

STATE_STALE_S = 0.2       # the board reports at 50 Hz: 10 missed reports


class BoardNotReady(RuntimeError):
    pass


class LiftLink:
    """Calls the lift and reads where it is: go2_world's car over UDP 9880,
    standing in for the person at the call button (or a building's lift API).
    The robot never waits on it for its own height: that comes from the board."""

    def __init__(self, host="127.0.0.1", port=9880, timeout=0.5):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def _ask(self, msg):
        self.sock.sendto(json.dumps(msg).encode(), self.addr)
        return json.loads(self.sock.recvfrom(1024)[0])

    def call(self, floor):
        return self._ask({"call": int(floor)})

    def state(self):
        return self._ask({})

    def close(self):
        self.sock.close()


class BoardStopped(RuntimeError):
    """The board left WALK by itself (a safety trip, or a person's limp):
    whatever the brain sends now, the dog is not walking it. The run ends
    here and says so, rather than tracking a waypoint the dog never nears."""
    layer = "go2_board"


class Go2BoardRobot(RobotInterface):
    CONTROL_DT  = 0.05         # 20 Hz, the twin's rate: the brain is tuned at it
    MAX_V       = 0.8          # m/s   -- the twin's limits. The Go2 policy is
    MAX_W       = 1.5          # rad/s    trained to +-1 rad/s: check before WALK
    CAM_FORWARD = 0.34         # m, lens ahead of body centre: measure on the dog

    def __init__(self, origin=(0.0, 0.0, 0.0), port=None, board_port=None,
                 clock=time.monotonic, sleep=time.sleep, board_host=None, listen=None):
        """`origin` is where the board's odometry frame starts, in the map
        frame: (x, y, yaw) of the dog when the board came up. FAST-LIO starts
        wherever it was switched on; the map is the building's.

        `board_host` / `listen`: the board's address and this machine's on the
        link between them, when the brain is not on the board (HIL over
        USB-C: 192.168.55.1 and 192.168.55.100). Localhost otherwise."""
        from go2_board.link import BOARD_PORT, BRAIN_PORT, HOST, BrainLink
        self.link = BrainLink(port=port or BRAIN_PORT, board_port=board_port or BOARD_PORT,
                              listen=listen or HOST, board_host=board_host or HOST)
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


# -- the board in run_building -------------------------------------------------

from cyberdog.sim.robot.mujoco_robot import MujocoRobot  # noqa: E402


class BoardTwinRobot(MujocoRobot):
    """go2_board drives; the twin follows it as a shadow.

    The pose comes from the board (go2_world's physics, or the dog). Every
    tick the twin is put there, so the camera, the virtual LiDAR, the people
    and the ground-truth checks run_building reads from model/data see the
    building from where the dog really is. Floor 1 only: the lift ride and
    the stairs are scripted moves (place() somewhere else), which a real
    board cannot do.
    """
    MAX_W = 1.0               # rad/s: go2_flat_v5 is trained to +-1
    REAL_TIME = True          # the board walks on the wall clock: never block the loop
    LIMP_HOLD = 4             # 20 Hz ticks of "limp" between two stands: 0.2 s
    # place() on the board is a pose to walk to, not one to be put at: the
    # scripted moves in and out of the lift car become velocity commands that
    # chase the moving target. Gains and limits for that.
    TRACK_K, TRACK_KW = 1.5, 2.0
    TRACK_V, TRACK_VY, TRACK_W = 0.5, 0.2, 0.8
    # The lift ride: arrived when the board's own height has settled this near
    # the new floor plus the standing height it rode at, for SETTLE_S.
    ARRIVED_M, SETTLE_S, RIDE_TIMEOUT_S = 0.05, 1.0, 60.0

    CLOUD_STALE_S = 0.5       # an older cloud is no cloud: the brain is blind, as the board is

    def __init__(self, scene_xml, start_z=0.0, board=None, origin=(0.0, 0.0, 0.0),
                 board_host=None, listen=None, lift=None, clouds=True):
        self.board = board or Go2BoardRobot(origin=origin, board_host=board_host, listen=listen)
        self.lift = lift          # LiftLink, for rides; None: no floor but the first
        # The Mid-360 as the board sees it, forwarded on 9872: what the brain
        # perceives with, as on the dog. clouds=False keeps the twin's own
        # raycast (it sees only what the twin's scene has, never a real person).
        self.clouds = None
        if clouds:
            from go2_board.lidar import BRAIN_CLOUD_PORT, CloudLink
            from go2_board.link import HOST
            self.clouds = CloudLink(port=BRAIN_CLOUD_PORT, host=listen or HOST)
        x, y, yaw = self._wait_for_pose()
        super().__init__(scene_xml, start_xy=(x, y), start_yaw=yaw, start_z=start_z)

    def _wait_for_pose(self, timeout=5.0):
        end = self.board.clock() + timeout
        while True:
            try:
                return self.board.get_pose()
            except BoardNotReady:
                if self.board.clock() > end:
                    raise
                self.board.sleep(self.board.CONTROL_DT)

    def start(self, timeout=5.0, tries=3):
        """Stand, then walk in place: what the board needs before velocities.

        A stand clears the board's safety trip once, when it is asked; if the
        board trips again straight away (a stale state as it wakes), asking
        "stand" again is no new request. So go limp and ask afresh. The limp
        is held for LIMP_HOLD ticks: the board is limp already, and keeps only
        the newest datagram, so a single one followed at once by a stand
        would never be seen."""
        for k in range(tries):
            try:
                self.board.stand(timeout=timeout)
                break
            except BoardNotReady:
                if k == tries - 1:
                    raise
                for _ in range(self.LIMP_HOLD):
                    self.board.request("limp")
                    self.board.sleep(self.board.CONTROL_DT)
        self.board.request("walk", until="WALK", timeout=timeout)

    def set_velocity(self, vx, vy=0.0, w=0.0):
        super().set_velocity(vx, vy, w)
        self.board.set_velocity(self._vx, self._vy, self._w)

    def step(self):
        self.board.step()                    # send, then wait for the 20 Hz tick
        self._follow()

    def place(self, xy, yaw, z=None):
        """One tick walking toward (xy, yaw): the board cannot be put anywhere.
        A height other than the floor's is a scripted climb (stairs), refused;
        the lift goes through ride_lift()."""
        if z is not None and abs(z - self.z) > 1e-6:
            raise NotImplementedError("a scripted climb: the board rides the lift, see ride_lift")
        x, y, th = self.get_pose()
        ex, ey = xy[0] - x, xy[1] - y
        c, s = math.cos(th), math.sin(th)
        fwd, side = c * ex + s * ey, -s * ex + c * ey
        turn = math.atan2(math.sin(yaw - th), math.cos(yaw - th))
        clip = lambda v, m: max(-m, min(m, v))
        self.board.set_velocity(clip(self.TRACK_K * fwd, self.TRACK_V),
                                clip(self.TRACK_K * side, self.TRACK_VY),
                                clip(self.TRACK_KW * turn, self.TRACK_W))
        self.board.step()
        self._follow()

    def ride_lift(self, to_floor, floor_z, on_tick=lambda z: None):
        """Stand in the car while the lift carries the dog to `to_floor`.

        What the dog does on the real lift: stop walking, stand, and wait
        until it is on the new floor, judged by its own height (the board's
        pose; on the dog, FAST-LIO's), then walk again. The call is the
        person's (here the lift link, standing in for them). `on_tick(z)` is
        called every tick with the car floor's height, for the video."""
        if self.lift is None:
            raise NotImplementedError("no lift link: --robot go2_board needs go2_world's lift")
        self.board.set_velocity(0.0, 0.0, 0.0)
        self.board.request("stand", until="STAND", timeout=5.0)
        s = self.board.board_state()
        stand_h = s["z"] - self.z              # trunk above the floor, standing
        self.lift.call(to_floor)
        goal = floor_z + stand_h
        start, settled = self.board.clock(), None
        while True:
            self.board.request("stand")        # one tick: the board hears from us at 20 Hz
            s = self.board.board_state()
            if s is None:
                raise BoardNotReady("no state from the board during the ride")
            if s["mode"] != "STAND":
                raise BoardStopped(f"the board is {s['mode']} in the lift"
                                   + (f": {s['tripped']}" if s.get("tripped") else ""))
            floor_now = s["z"] - stand_h
            x, y, th = self.board.get_pose()
            MujocoRobot.place(self, (x, y), th, z=floor_now)
            on_tick(floor_now)
            now = self.board.clock()
            if abs(s["z"] - goal) < self.ARRIVED_M:
                settled = settled or now
                if now - settled >= self.SETTLE_S:
                    break
            else:
                settled = None
            if now - start > self.RIDE_TIMEOUT_S:
                raise BoardStopped(f"the lift did not reach floor {to_floor} in "
                                   f"{self.RIDE_TIMEOUT_S:.0f} s (board at z {s['z']:.2f})")
            self.board.sleep(self.CONTROL_DT)
        self.z = floor_z
        self.board.request("walk", until="WALK", timeout=5.0)

    def _follow(self):
        x, y, yaw = self.board.get_pose()
        MujocoRobot.place(self, (x, y), yaw)
        s = self.board.board_state()
        if s["mode"] != "WALK":
            why = f": {s['tripped']}" if s.get("tripped") else ""
            raise BoardStopped(f"the board is {s['mode']}, not walking{why}")

    def now(self):
        return self.board.now()

    def get_scan(self):
        """The board's newest Mid-360 cloud in the map frame, (N, 3).

        The board sends it levelled, in the trunk frame, the floor at z = 0;
        turned by the dog's heading and set on the floor it is on. Empty when
        no cloud has come for CLOUD_STALE_S: blind, which perception treats as
        seeing nothing, while the board itself stops translation."""
        if self.clouds is None:
            return MujocoRobot.get_scan(self)
        c = self.clouds.poll()
        if c is None or self.board.clock() - self._cloud_seen(c) > self.CLOUD_STALE_S:
            return np.empty((0, 3))
        x, y, th = self.get_pose()
        cs, sn = math.cos(th), math.sin(th)
        p = c.xyz.astype(float)
        return np.column_stack((x + cs * p[:, 0] - sn * p[:, 1],
                                y + sn * p[:, 0] + cs * p[:, 1],
                                self.z + p[:, 2]))

    def _cloud_seen(self, c):
        """When this cloud first arrived here, on this side's clock."""
        if c.seq != getattr(self, "_cloud_seq", None):
            self._cloud_seq, self._cloud_at = c.seq, self.board.clock()
        return self._cloud_at

    def close(self):
        self.board.close()
        if self.lift is not None:
            self.lift.close()
        if self.clouds is not None:
            self.clouds.sock.close()
