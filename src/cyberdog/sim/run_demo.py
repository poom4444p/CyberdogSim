"""Drive the dog along a planned route -- the closed loop, no video yet.

Pure pursuit standing in for VAMOS: the projector picks a destination 2-4 m ahead,
we steer at it. VAMOS replaces the steering later; everything around it stays.
"""
import argparse
import math


from cyberdog import paths
from cyberdog.planning.checkpoint_projector import load_camera_config, project_route
from cyberdog.sim.control import GOAL_R, K_W, TURN_ONLY, advance, plan_route, wrap
from cyberdog.sim.robot.mujoco_robot import MujocoRobot


def drive(robot, waypoints, cam, max_steps=4000, verbose=True):
    """Follow the waypoints. Returns (reached, steps)."""
    i = 1 if len(waypoints) > 1 else 0      # waypoint 0 is where we start
    for n in range(max_steps):
        x, y, yaw = robot.get_pose()

        i = advance(i, x, y, waypoints)

        goal = waypoints[-1]
        if math.hypot(goal[0] - x, goal[1] - y) < GOAL_R:
            return True, n

        # Steer at the checkpoint itself, not at a destination beyond it. A
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
    robot = MujocoRobot(str(paths.FLOOR1_SCENE),
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
