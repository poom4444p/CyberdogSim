"""Drawing onto the dog's camera frame -- goal pixel, paths, LiDAR returns.

Split out of `record_demo.py` for the same reason `clearance.py` was: both
`record_demo.py` and `run_building.py` render these overlays, so the second one
imported the first just to draw a crosshair, and the video recorder read as a
library with a `main()` attached.

Everything here mutates and returns `img`, a (H, W, 3) uint8 array. There is no
cv2 in this environment, so it is all numpy slicing -- which is also why the
shapes are rectangles and 1-pixel-thick lines rather than anything antialiased.

The overlays are not decoration. `draw_points` is what makes the swerve legible:
without the unexplained LiDAR returns on screen, the video shows a dog veering
across a corridor for no visible reason, and there is no way to tell a correct
avoidance from a control bug.
"""
import numpy as np

from cyberdog.planning.checkpoint_projector import project_to_pixel


def draw_marker(img, px, py, colour=(255, 80, 80), r=9):
    """Crosshair and ring at the goal pixel."""
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    img[(d > r - 1.5) & (d < r + 1.5)] = colour
    img[max(py - 14, 0):min(py + 15, h), max(px - 1, 0):min(px + 2, w)] = colour
    img[max(py - 1, 0):min(py + 2, h), max(px - 14, 0):min(px + 15, w)] = colour
    return img


def safety_bar(img, factor):
    """The safety factor as a short bar in the corner of the dog's view.

    Green when the dog is confident it can walk the path it chose, amber as it
    approaches the gate. A number nobody can see is a number nobody checks.
    """
    h = img.shape[0]
    w = int(70 * max(0.0, min(1.0, factor)))
    colour = (90, 200, 120) if factor > 0.75 else (235, 180, 70)
    img[h - 26:h - 14, 12:12 + 70] = (40, 42, 48)
    if w:
        img[h - 24:h - 16, 14:14 + max(w - 4, 1)] = colour
    return img


def bar(img, frac, colour=(90, 200, 120)):
    """Thin progress strip along the bottom."""
    w = img.shape[1]
    img[-6:, :int(w * frac)] = colour
    return img


def draw_path(img, pts, pose, cam, colour, thick=1):
    """Draw a map-frame path back into the camera image."""
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
                img[max(yy - thick, 0):yy + thick + 1,
                    max(xx - thick, 0):xx + thick + 1] = colour
        prev = (u, v)
    return img


def draw_points(img, pts, pose, cam, colour=(255, 140, 40), r=2):
    """Map-frame points back into the camera image -- the LiDAR returns the
    static map could not account for.

    Without these on screen the video shows a dog swerving for no visible
    reason. They are what it swerved for.
    """
    h, w = img.shape[:2]
    for p in pts:
        res = project_to_pixel((float(p[0]), float(p[1])), pose, cam)
        if res["state"] != "TRACK":
            continue
        u, v = res["pixel"]
        if 0 <= u < w and 0 <= v < h:
            img[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1] = colour
    return img
