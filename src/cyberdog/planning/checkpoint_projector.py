"""Checkpoint Projector (Task Planner layer).

Takes A* checkpoints (map frame, meters), the robot pose, and camera
intrinsics, and returns the pixel VAMOS should aim for -- or ALIGN when the
destination point isn't visible. Hardware-free: uses the same pinhole math as
VAMOS/vamos_ws/src/vamos/nodes/navigate.py:project_point_to_image, but with
a level camera assumed and the pose passed in instead of read from ROS tf.

Usage:
    python3 checkpoint_projector.py   # runs the sanity checks below

Running on Linux:
    Only needs PyYAML (pip install pyyaml). Works from any directory:
        python -m cyberdog.planning.checkpoint_projector
"""
import math
import yaml

from cyberdog.paths import CAMERA_CONFIG as DEFAULT_CONFIG


def load_camera_config(path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_destination(robot_xy, waypoints, min_d=2.0, max_d=4.0):
    """Pick a point 2-4 m ahead along the route.

    Returns the first waypoint in [min_d, max_d]. If the route jumps from
    closer than min_d to farther than max_d, interpolate along that segment
    so the destination lands at max_d. If the whole remaining route is within
    min_d, return the last waypoint (we're nearly there).
    """
    rx, ry = robot_xy
    prev = (rx, ry)
    for wp in waypoints:
        d = math.hypot(wp[0] - rx, wp[1] - ry)
        if d < min_d:
            prev = wp
            continue
        if d <= max_d:
            return tuple(wp)
        # Segment prev -> wp crosses the max_d circle; binary search for it.
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            px = prev[0] + mid * (wp[0] - prev[0])
            py = prev[1] + mid * (wp[1] - prev[1])
            if math.hypot(px - rx, py - ry) < max_d:
                lo = mid
            else:
                hi = mid
        return (prev[0] + lo * (wp[0] - prev[0]), prev[1] + lo * (wp[1] - prev[1]))
    return tuple(waypoints[-1]) if waypoints else None


def project_to_pixel(point_xy, robot_pose, cam):
    """Project a floor point (map frame) into the camera image.

    robot_pose: (x, y, yaw) with yaw in radians, CCW from the map x axis.
    Returns {"state": "TRACK", "pixel": (u, v), "loc": (loc_x, loc_y)} or
            {"state": "ALIGN", "turn": "left"|"right"}.
    """
    rx, ry, yaw = robot_pose
    dx, dy = point_xy[0] - rx, point_xy[1] - ry

    # Map frame -> robot frame (x forward, y left)
    x_fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
    y_left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    turn = "left" if y_left > 0 else "right"

    # Robot frame -> camera frame (OpenCV: z forward, x right, y down).
    # Camera assumed level; goal is on the floor, camera_height below the lens.
    z_cam = x_fwd
    x_cam = -y_left
    y_cam = cam["camera_height"]

    if z_cam <= 0:
        return {"state": "ALIGN", "turn": turn}

    u = cam["fx"] * x_cam / z_cam + cam["cx"]
    v = cam["fy"] * y_cam / z_cam + cam["cy"]

    m = cam.get("fov_margin", 0)
    if not (m <= u < cam["width"] - m and m <= v < cam["height"] - m):
        return {"state": "ALIGN", "turn": turn}

    # PaliGemma location tokens: 1024 bins (<loc0000>..<loc1023>) across the
    # image. Same scale VAMOS decodes with (vamos_demo.py: loc * width / 1024),
    # clamped so a pixel on the far edge doesn't become <loc1024>.
    loc_x = min(int(u / cam["width"] * 1024), 1023)
    loc_y = min(int(v / cam["height"] * 1024), 1023)
    return {"state": "TRACK", "pixel": (int(u), int(v)), "loc": (loc_x, loc_y)}


def project_route(robot_pose, waypoints, cam, min_d=2.0, max_d=4.0, pick_from=None):
    """Convenience wrapper: pick the destination, then project it.

    `pick_from` is where along the route the destination is measured from,
    when that is not where the lens is. It matters: the lens sits
    CAM_FORWARD ahead of the body, so with the pose alone the choice of
    destination turns with the dog, and a waypoint sitting near the min_d
    boundary hops in and out of range as the body rotates. The dog then steers
    alternately at that waypoint and at a point 4 m down the leg past it, both
    far enough off its heading to be a turn rather than a step -- and a turn
    does not move it, so the geometry never changes and it pivots on the spot
    until the leg times out. Measured from the body it is the same 2-4 m
    whichever way the dog happens to be looking. The projection itself still
    comes from `robot_pose`, because that is what the camera can see.
    """
    destination = pick_destination(pick_from if pick_from is not None else robot_pose[:2],
                                   waypoints, min_d, max_d)
    if destination is None:
        return {"state": "ARRIVED"}
    result = project_to_pixel(destination, robot_pose, cam)
    result["destination"] = destination
    return result


def vamos_prompt(result):
    """Format a TRACK result as a VAMOS text prompt."""
    lx, ly = result["loc"]
    return f"Navigate to x=<loc{lx:04d}>, y=<loc{ly:04d}>."


if __name__ == "__main__":
    cam = load_camera_config()

    def check(name, got, expect_state, **extra):
        ok = got["state"] == expect_state and all(got.get(k) == v for k, v in extra.items())
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {got}")

    # 1. Straight ahead 3 m: u at image center, v below center
    r = project_to_pixel((3, 0), (0, 0, 0), cam)
    check("straight ahead", r, "TRACK")
    assert r["pixel"][0] == int(cam["cx"]) and r["pixel"][1] > cam["cy"]

    # 2. Directly left: behind/at the image plane -> ALIGN left
    check("directly left", project_to_pixel((0, 3), (0, 0, 0), cam), "ALIGN", turn="left")

    # 3. Behind the robot -> ALIGN
    check("behind", project_to_pixel((-3, 0), (0, 0, 0), cam), "ALIGN")

    # 4. Same left point, robot rotated 90 deg to face it -> straight ahead
    r = project_to_pixel((0, 3), (0, 0, math.pi / 2), cam)
    check("rotated to face it", r, "TRACK")
    assert r["pixel"][0] == int(cam["cx"])

    # 5. Slightly right of forward -> TRACK, u right of center
    r = project_to_pixel((3, -0.5), (0, 0, 0), cam)
    check("slightly right", r, "TRACK")
    assert r["pixel"][0] > cam["cx"]

    # 6. Destination picking: interpolate onto the 4 m circle
    c = pick_destination((0, 0), [(1, 0), (10, 0)])
    print(f"[{'PASS' if abs(c[0] - 4.0) < 1e-3 else 'FAIL'}] destination interpolation: {c}")

    # 7. <loc> tokens round-trip through VAMOS's decoding (loc * size / 1024)
    r = project_to_pixel((4, 1.2), (0, 0, 0), cam)
    back = (r["loc"][0] * cam["width"] / 1024, r["loc"][1] * cam["height"] / 1024)
    ok = all(abs(b - p) <= max(cam["width"], cam["height"]) / 1024 for b, p in zip(back, r["pixel"]))
    print(f"[{'PASS' if ok else 'FAIL'}] loc round-trip: pixel {r['pixel']} -> {r['loc']} -> VAMOS pixel "
          f"({back[0]:.1f}, {back[1]:.1f})")

    print("\nExample prompt:", vamos_prompt(project_route((0, 0, 0), [(1, 0), (3, 0.2), (6, 0)], cam)))
