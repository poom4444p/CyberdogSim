"""Drive the dog along a planned route -- the closed loop, no video yet.

Pure pursuit standing in for VAMOS: the projector picks a carrot 2-4 m ahead,
we steer at it. VAMOS replaces the steering later; everything around it stays.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "planner_layer"))

from building_router import BuildingRouter
from checkpoint_projector import load_camera_config, pick_carrot, project_route
from mujoco_robot import MujocoRobot

START_XY = (45.0, 4.0)          # main entrance, floor 1
ARRIVE_R = 0.15                 # waypoint reached inside this -- keep it well
                                # under half a door width (doors are 0.9 m) or the
                                # dog turns early and clips the frame

GOAL_R = 0.5
K_W = 2.0                       # rad/s per rad of heading error
TURN_ONLY = 0.8                 # above this error, pivot instead of walking


def advance(i, x, y, waypoints):
    """Move to the next waypoint once we are near it or have driven past it.

    Distance alone is not enough: overshoot a waypoint by more than ARRIVE_R
    and the dog turns back for it, overshoots again, and spins forever.
    Projecting onto the segment catches the overshoot.
    """
    while i < len(waypoints) - 1:
        px, py = waypoints[i - 1]
        cx, cy = waypoints[i]
        sx, sy = cx - px, cy - py
        seg = sx * sx + sy * sy
        t = ((x - px) * sx + (y - py) * sy) / seg if seg > 1e-9 else 1.0
        if t >= 1.0 or math.hypot(cx - x, cy - y) < ARRIVE_R:
            i += 1
        else:
            break
    return i


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan_route(destination, start_xy=START_XY, floor=1):
    router = BuildingRouter()
    target = router.resolve(destination, floor, start_xy)
    if target is None:
        raise SystemExit(f"unknown destination: {destination}")
    legs = router.plan(floor, start_xy, target)
    if not legs:
        raise SystemExit("no route found")
    return [(float(cp.position[0]), float(cp.position[1]))
            for _, route in legs for cp in route.checkpoints]


def drive(robot, waypoints, cam, max_steps=4000, verbose=True):
    """Follow the waypoints. Returns (reached, steps)."""
    i = 1 if len(waypoints) > 1 else 0      # waypoint 0 is where we start
    for n in range(max_steps):
        x, y, yaw = robot.get_pose()

        i = advance(i, x, y, waypoints)

        goal = waypoints[-1]
        if math.hypot(goal[0] - x, goal[1] - y) < GOAL_R:
            return True, n

        # Steer at the checkpoint itself, not at a carrot beyond it. A carrot
        # lookahead cuts corners, and a corner here is a 0.9 m doorway.
        goal_i = waypoints[i]
        err = wrap(math.atan2(goal_i[1] - y, goal_i[0] - x) - yaw)

        w = K_W * err
        vx = 0.0 if abs(err) > TURN_ONLY else robot.MAX_V * math.cos(err)
        robot.set_velocity(vx, 0.0, w)
        robot.step()

        if verbose and n % 40 == 0:
            state = project_route(robot.camera_pose(), waypoints[i:], cam)
            px = state.get("pixel", "-")
            print(f"  t={robot.sim_time:6.1f}s  pos=({x:5.1f},{y:5.1f})  "
                  f"yaw={math.degrees(yaw):7.1f}  wp={i}/{len(waypoints)-1}  "
                  f"{state['state']:5s} pixel={px}")
    return False, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("destination", nargs="?", default="restroom")
    args = ap.parse_args()

    waypoints = plan_route(args.destination)
    print(f"route to {args.destination}: {len(waypoints)} checkpoints")
    for p in waypoints:
        print(f"    ({p[0]:5.1f}, {p[1]:5.1f})")

    # Face the first leg so it doesn't open with a pivot.
    yaw0 = math.atan2(waypoints[1][1] - waypoints[0][1],
                      waypoints[1][0] - waypoints[0][0])
    robot = MujocoRobot(os.path.join(HERE, "scene_cache", "floor1.xml"),
                        start_xy=waypoints[0], start_yaw=yaw0)
    cam = load_camera_config()

    print("driving:")
    reached, steps = drive(robot, waypoints, cam)
    x, y, _ = robot.get_pose()
    print(f"{'ARRIVED' if reached else 'FAILED '} after {steps} steps "
          f"({robot.sim_time:.1f}s sim) at ({x:.2f}, {y:.2f}); "
          f"goal ({waypoints[-1][0]:.2f}, {waypoints[-1][1]:.2f})")


if __name__ == "__main__":
    main()
