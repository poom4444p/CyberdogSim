"""run_building with traces, for the HIL rig: every 4th velocity command, place()
target and lift tick, with the board's mode, trip and LiDAR front clearance.

  python trace_run_building.py DEST [run_building's arguments]   (onboard.sh brain, TRACE=1)

A line reads: CMD t=<sim s> pose (x,y,yaw) z <body height> | vx,vy,w | board <mode>
trip <reason> F <front clearance m>. These lines are what showed, on the J4012,
the dog commanded to walk while its pose did not move.
"""
import runpy
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
              f"board {s.get('mode')} trip {s.get('tripped')} F {lidar.get('front')}", flush=True)


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
sys.argv = ["run_building"] + sys.argv[1:]
runpy.run_module("cyberdog.sim.run_building", run_name="__main__")
