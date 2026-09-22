"""Record the run as an MP4: chase view + what the dog sees.

The left panel is for people, the right panel is the actual input to the
navigation stack -- same 640x480 frame VAMOS will get, with the projected
goal pixel drawn on it.
"""
import argparse
import math
import os
import sys

import imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "planner_layer"))

from build_scene import load_grid
from checkpoint_projector import load_camera_config, project_route, project_to_pixel, vamos_prompt
from vamos_client import VamosPolicy
from mujoco_robot import MujocoRobot
from run_demo import ARRIVE_R, GOAL_R, K_W, TURN_ONLY, advance, plan_route, wrap

FPS = 20                      # equals the 20 Hz control rate, so video is real time
REPLAN_EVERY = 20             # ticks between VLM calls -- 1 Hz, near VAMOS's own RHC rate
FOLLOW_D = 1.5                # how far along the chosen path to aim, metres


def draw_marker(img, px, py, colour=(255, 80, 80), r=9):
    """Crosshair and ring at the goal pixel. No cv2 in this env, so numpy."""
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    img[(d > r - 1.5) & (d < r + 1.5)] = colour
    img[max(py - 14, 0):min(py + 15, h), max(px - 1, 0):min(px + 2, w)] = colour
    img[max(py - 1, 0):min(py + 2, h), max(px - 14, 0):min(px + 15, w)] = colour
    return img


def bar(img, frac, colour=(90, 200, 120)):
    """Thin progress strip along the bottom."""
    w = img.shape[1]
    img[-6:, :int(w * frac)] = colour
    return img


def free_space_test(robot_radius=0.16):
    """(x, y) -> is this clear on the ground-truth map? Stands in for the MLP."""
    occ, res, ox, oy = load_grid(1)
    H, W = occ.shape

    def is_free(x, y):
        c0 = int((x - robot_radius - ox) / res); c1 = int((x + robot_radius - ox) / res)
        r1 = int(H - 1 - (y - robot_radius - oy) / res)
        r0 = int(H - 1 - (y + robot_radius - oy) / res)
        if c0 < 0 or r0 < 0 or c1 >= W or r1 >= H:
            return False
        return not occ[r0:r1 + 1, c0:c1 + 1].any()
    return is_free


def path_target(path, xy, d=FOLLOW_D):
    """First point on the chosen path at least d metres away, else its end."""
    for p in path:
        if math.dist(p, xy) >= d:
            return p
    return path[-1]


def draw_path(img, pts, pose, cam, colour, thick=1):
    """Draw a map-frame path back into the camera image."""
    h, w = img.shape[:2]
    prev = None
    for p in pts:
        r = project_to_pixel(p, pose, cam)
        if r["state"] != "TRACK":
            prev = None
            continue
        u, v = r["pixel"]
        if prev is not None:
            n = max(abs(u - prev[0]), abs(v - prev[1]), 1)
            for k in range(n + 1):
                xx = int(prev[0] + (u - prev[0]) * k / n)
                yy = int(prev[1] + (v - prev[1]) * k / n)
                img[max(yy - thick, 0):yy + thick + 1, max(xx - thick, 0):xx + thick + 1] = colour
        prev = (u, v)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("destination", nargs="?", default="restroom")
    ap.add_argument("--out", default=os.path.join(HERE, "scene_cache", "demo.mp4"))
    ap.add_argument("--vamos", action="store_true",
                    help="steer with the VLM instead of straight at the checkpoint")
    args = ap.parse_args()

    waypoints = plan_route(args.destination)
    yaw0 = math.atan2(waypoints[1][1] - waypoints[0][1],
                      waypoints[1][0] - waypoints[0][0])
    robot = MujocoRobot(os.path.join(HERE, "scene_cache", "floor1.xml"),
                        start_xy=waypoints[0], start_yaw=yaw0)
    cam = load_camera_config()

    chase = mujoco.MjvCamera()
    chase.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.distance, chase.elevation = 4.5, -28
    outside = mujoco.Renderer(robot.model, 480, 640)

    total = sum(math.dist(waypoints[k], waypoints[k + 1])
                for k in range(len(waypoints) - 1))

    policy = chosen = None
    candidates = []
    if args.vamos:
        policy = VamosPolicy(cam, free_space_test())
        if not policy.available():
            raise SystemExit("VAMOS server is not answering on 127.0.0.1:8009 -- "
                             "start VAMOS/server/vlm_server.py first")
        print("VAMOS in the loop, replanning every "
              f"{REPLAN_EVERY / FPS:.1f}s of sim time")

    writer = imageio.get_writer(args.out, fps=FPS, macro_block_size=1)
    i, frames = 1, 0
    for n in range(4000):
        x, y, yaw = robot.get_pose()
        i = advance(i, x, y, waypoints)
        done = math.hypot(waypoints[-1][0] - x, waypoints[-1][1] - y) < GOAL_R

        state = project_route(robot.camera_pose(), waypoints[i:], cam)
        dog = robot.get_image().copy()

        # Ask the VLM on a slow clock; a call costs seconds, control runs at 20 Hz.
        if policy is not None and state["state"] == "TRACK" and n % REPLAN_EVERY == 0:
            chosen, candidates = policy.plan(
                dog, vamos_prompt(state), robot.camera_pose(), state["carrot"])

        cpose = robot.camera_pose()
        for c in candidates:
            draw_path(dog, c, cpose, cam, (120, 120, 130))
        if chosen:
            draw_path(dog, chosen, cpose, cam, (90, 220, 120), thick=2)
        if state["state"] == "TRACK":
            draw_marker(dog, *state["pixel"])

        chase.lookat[:] = [x, y, 0.3]
        chase.azimuth = math.degrees(yaw) + 180
        outside.update_scene(robot.data, chase)

        left = np.ascontiguousarray(outside.render())
        walked = sum(math.dist(waypoints[k], waypoints[k + 1]) for k in range(i - 1)) \
            + math.hypot(x - waypoints[i - 1][0], y - waypoints[i - 1][1])
        bar(left, min(walked / total, 1.0))
        writer.append_data(np.hstack([left, dog]))
        frames += 1

        if done:
            for _ in range(FPS):          # hold the last frame a second
                writer.append_data(np.hstack([left, dog]))
            break

        # VAMOS decides HOW when it has a usable path; the map still decides
        # WHERE, and alignment turns stay with the deterministic rules.
        if chosen and state["state"] == "TRACK":
            tx, ty = path_target(chosen, (x, y))
        else:
            tx, ty = waypoints[i]
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        robot.set_velocity(0.0 if abs(err) > TURN_ONLY else robot.MAX_V * math.cos(err),
                           0.0, K_W * err)
        robot.step()

    writer.close()
    print(f"{'arrived' if done else 'timed out'} -- {frames} frames, "
          f"{frames / FPS:.1f}s of video -> {args.out}")
    if policy is not None:
        print(f"VAMOS: {policy.stats['calls']} calls, "
              f"{policy.stats['failures']} failed, "
              f"{policy.stats['rejected']} candidate paths rejected by the gate")


if __name__ == "__main__":
    main()
