"""Record the run as an MP4: chase view + what the dog sees.

The left panel is for people, the right panel is the actual input to the
navigation stack -- same 640x480 frame VAMOS will get, with the projected
goal pixel drawn on it.

Single floor only, and that is the point of it existing alongside
`run_building.py`: this is the shortest closed loop that produces a video, so
when something looks wrong in the full three-storey run it is the thing to fall
back to. The clearance field, the drawing helpers and the control law it used to
own now live in `clearance.py`, `overlay.py` and `control.py`, because the live
stack needs them too and should not have to import a recorder to get them.

    python -m cyberdog.sim.record_demo restroom
    python -m cyberdog.sim.record_demo restroom --vamos    # VLM in the loop
"""
import argparse
import math

import imageio
import mujoco
import numpy as np

from cyberdog import paths
from cyberdog.planning.checkpoint_projector import (load_camera_config,
                                                    project_route, vamos_prompt)
from cyberdog.sim.clearance import clearance_test, free_space_test
from cyberdog.sim.control import (GOAL_R, K_W, TURN_ONLY, advance, path_target,
                                  plan_route, wrap)
from cyberdog.sim.overlay import bar, draw_marker, draw_path, safety_bar
from cyberdog.sim.robot.mujoco_robot import MujocoRobot
from cyberdog.sim.vamos_client import VamosPolicy

FPS = 20                      # equals the 20 Hz control rate, so video is real time
REPLAN_EVERY = 20             # ticks between VLM calls -- 1 Hz, near VAMOS's own RHC rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("destination", nargs="?", default="restroom")
    ap.add_argument("--out", default=str(paths.SCENE_CACHE / "demo.mp4"))
    ap.add_argument("--vamos", action="store_true",
                    help="steer with the VLM instead of straight at the checkpoint")
    args = ap.parse_args()
    paths.ensure_output()

    waypoints = plan_route(args.destination)
    yaw0 = math.atan2(waypoints[1][1] - waypoints[0][1],
                      waypoints[1][0] - waypoints[0][0])
    robot = MujocoRobot(str(paths.FLOOR1_SCENE),
                        start_xy=waypoints[0], start_yaw=yaw0)
    cam = load_camera_config()

    chase = mujoco.MjvCamera()
    chase.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.distance, chase.elevation = 4.5, -28
    outside = mujoco.Renderer(robot.model, 480, 640)

    total = sum(math.dist(waypoints[k], waypoints[k + 1])
                for k in range(len(waypoints) - 1))

    policy = chosen = None
    safety = 1.0
    candidates = []
    if args.vamos:
        from cyberdog.sim.sensing.dreaming import Dream
        policy = VamosPolicy(cam, free_space_test(1),
                             dream=Dream(clearance_test(1), robot.MAX_V))
        if not policy.available():
            raise SystemExit("VAMOS server is not answering on 127.0.0.1:8009 -- "
                             "start vendor/VAMOS/server/vlm_server.py first")
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
            chosen, candidates, safety = policy.plan(
                dog, vamos_prompt(state), robot.camera_pose(), state["destination"],
                pose=(x, y, yaw))

        cpose = robot.camera_pose()
        for c in candidates:
            draw_path(dog, c, cpose, cam, (120, 120, 130))
        if chosen:
            draw_path(dog, chosen, cpose, cam, (90, 220, 120), thick=2)
        if state["state"] == "TRACK":
            draw_marker(dog, *state["pixel"])
        if chosen:
            safety_bar(dog, safety)

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
        # A path the dog only just believes it can walk is walked at half
        # pace; the map's own route is the trusted fallback and is not
        # throttled. See dreaming.py for where the factor comes from.
        v = robot.MAX_V * (safety if (chosen and state["state"] == "TRACK") else 1.0)
        robot.set_velocity(0.0 if abs(err) > TURN_ONLY else v * math.cos(err),
                           0.0, K_W * err)
        robot.step()

    writer.close()
    print(f"{'arrived' if done else 'timed out'} -- {frames} frames, "
          f"{frames / FPS:.1f}s of video -> {args.out}")
    if policy is not None:
        st = policy.stats
        mean = st["safety_sum"] / st["chosen"] if st["chosen"] else 0.0
        print(f"VAMOS: {st['calls']} calls, {st['failures']} failed, "
              f"{st['rejected']} candidates rejected "
              f"({st['rejected_by_gate']} of them by the safety gate), "
              f"mean safety of the paths it followed {mean:.2f}")


if __name__ == "__main__":
    main()
