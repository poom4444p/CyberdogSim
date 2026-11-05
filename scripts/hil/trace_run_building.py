"""run_building with traces, for the HIL rig: every 4th velocity command, place()
target and lift tick, with the board's mode, trip and LiDAR front clearance.

  python trace_run_building.py DEST [run_building's arguments]   (onboard.sh brain, TRACE=1)

A line reads: CMD t=<sim s> pose (x,y,yaw) z <body height> | vx,vy,w | board <mode>
trip <reason> F <front> L <left> R <right>, the board guard's free distances in m.
These lines are what showed, on the J4012, the dog commanded to walk while its
pose did not move.

And a BOARD line on every tick the guard changes the plan's command
(Run.respect_board): the phase (side, turn, gave up), what the plan asked, what
went to the board, and the guard's free distances.
"""
import sys

from cyberdog.sim.robot import go2_board_robot as g

EVERY = 4
n = [0]


def show(self, tag, extra=""):
    n[0] += 1
    if n[0] % EVERY == 0:
        s = self.board._state or {}
        lidar = s.get("lidar") or {}
        x, y, th = self.get_pose()
        print(f"  {tag} t={self.now():6.1f} pose ({x:.2f},{y:.2f},{th:+.2f}) z {s.get('z') or 0:.2f} | {extra} | "
              f"board {s.get('mode')} trip {s.get('tripped')} {free(lidar)}", flush=True)


def free(lidar):
    """The board guard's free distances, m: front, left, right (- where it sees nothing)."""
    f = lambda k: "-" if lidar.get(k) is None else f"{lidar[k]:.2f}"   # noqa: E731
    return f"F {f('front')} L {f('left')} R {f('right')}"


orig_v, orig_p = g.BoardTwinRobot.set_velocity, g.BoardTwinRobot.place
orig_ride = g.BoardTwinRobot.ride_lift


def set_velocity(self, vx, vy=0.0, w=0.0):
    orig_v(self, vx, vy, w)
    show(self, "CMD", f"{vx:+.2f},{vy:+.2f},{w:+.2f}")


def place(self, xy, yaw, z=None):
    orig_p(self, xy, yaw, z)
    c = self.board._cmd
    show(self, "PLACE", f"to ({xy[0]:.2f},{xy[1]:.2f},{yaw:+.2f}) sent {c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}")


def ride_lift(self, to_floor, floor_z, on_tick=lambda z: None):
    def tick(z):
        show(self, "RIDE", f"car floor at {z:.2f}")
        on_tick(z)
    print(f"  RIDE start to floor {to_floor}", flush=True)
    r = orig_ride(self, to_floor, floor_z, tick)
    print(f"  RIDE done, z {self.z}", flush=True)
    return r


g.BoardTwinRobot.set_velocity, g.BoardTwinRobot.place = set_velocity, place
g.BoardTwinRobot.ride_lift = ride_lift

# Every tick the board's guard changes what the plan asked (Run.respect_board):
# what the plan wanted, what went out, and what the guard saw.
from cyberdog.sim import run_building as rb  # noqa: E402

orig_respect = rb.Run.respect_board


def respect_board(self, cmd, pose):
    out = orig_respect(self, cmd, pose)
    if tuple(out) != tuple(cmd):
        s = self.robot.board._state or {}
        x, y, th = pose
        phase = ("side" if self.board_side else "turn" if self.board_turning else
                 "gave up" if self.board_gave_up else "-")
        print(f"  BOARD t={self.robot.now():6.1f} pose ({x:.2f},{y:.2f},{th:+.2f}) {phase:7s} | "
              f"plan {cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f} -> {out[0]:+.2f},{out[1]:+.2f},{out[2]:+.2f} | "
              f"{free(s.get('lidar') or {})}", flush=True)
    return out


rb.Run.respect_board = respect_board
# main() of the module imported above, not runpy's fresh copy of it: in that
# copy Run is another class, and the patch above would not reach it.
sys.argv = ["run_building"] + sys.argv[1:]
rb.main()
