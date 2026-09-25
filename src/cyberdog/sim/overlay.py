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
import math

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


# -- the chase panel --------------------------------------------------------
#
# Everything above draws into the dog's own camera, using the pinhole model in
# checkpoint_projector. The chase view is a different camera -- a free camera
# MuJoCo places from azimuth/elevation/distance -- so it needs its own
# projection, and `ChaseCam` is it.
#
# The split matters: labels drawn here are a caption on the video and nothing
# more. They are not in the scene, so the dog's camera never contains them and
# VAMOS is never shown a room name. Putting signage in the XML instead would
# read the same on screen and be a different experiment.

# Where a door sign goes. Walls are 2.0 m (levels.FLOOR_HEIGHT is 2.2 with a
# 0.12 m slab), so anything near 1.95 m is up at the wall/ceiling junction and
# reads as a caption floating over the corridor rather than something on the
# wall by the door. Real signage sits at eye height.
LABEL_H = 1.55
LABEL_R = 8.0           # only name what the dog is passing; the corridor is
                        # 48 m long and labelling all of it is a wall of text
FADE_R = 2.0            # metres of fade-in at the far edge, so a name arrives
                        # rather than blinking on
# A sign gets bigger as you walk up to it. Fixed-size text does the opposite of
# what the eye expects -- a far name and a near name look equally important,
# and at 8 m down a corridor the near one is what you are trying to read.
SIGN_PX = 60            # font size is SIGN_PX / metres, clamped
SIGN_MIN, SIGN_MAX = 11, 24
PLATE = (24, 26, 32)    # the placard itself
EDGE = (150, 156, 170)  # its border, so it reads as a plate on a pale wall


class ChaseCam:
    """Where a world point lands in the chase panel.

    Built from `MjvScene.camera`, which MuJoCo fills in when the scene is
    updated -- so it describes the camera that actually rendered the frame
    rather than a second guess at the one we asked for.

    The two entries in that array are the left and right eyes of a stereo
    pair. Mono rendering is the midpoint between them: taking either one on
    its own puts every label a few pixels off centre, which is small enough to
    look like a bug in the projection rather than the wrong camera.
    """

    def __init__(self, scene, width, height):
        a, b = scene.camera[0], scene.camera[1]
        self.pos = (np.array(a.pos) + np.array(b.pos)) / 2.0
        self.fwd = np.array(a.forward, dtype=float)
        self.up = np.array(a.up, dtype=float)
        right = np.cross(self.fwd, self.up)
        self.right = right / (np.linalg.norm(right) or 1.0)
        self.near = float(a.frustum_near)
        self.half_h = float(a.frustum_top)
        self.half_w = self.half_h * (width / float(height))
        self.w, self.h = width, height

    def pixel(self, p):
        """(x, y, z) in map coordinates -> (px, py), or None if behind."""
        v = np.asarray(p, dtype=float) - self.pos
        z = float(v @ self.fwd)
        if z <= self.near:
            return None
        u = float(v @ self.right) * self.near / (self.half_w * z)
        w = float(v @ self.up) * self.near / (self.half_h * z)
        return int(round((u + 1.0) / 2.0 * self.w)), int(round((1.0 - w) / 2.0 * self.h))


def _font(size=13):
    """A bitmap font, at a size if this Pillow can do sizes.

    `load_default(size=)` arrived in Pillow 10.1 and pyproject only asks for
    9.5, so the fallback is the old fixed-size default rather than a crash on
    a valid install.
    """
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label_places(img, cam, places, dog_xy, floor_z, radius=LABEL_R):
    """Name the rooms the dog is passing, on the chase panel.

    `places` is (name, (x, y)) in map coordinates -- door positions, not room
    centres: the door is what the dog goes past and what a viewer is trying to
    match the voice line to. A room centre sits behind a wall and its label
    would float in the middle of a solid.

    Drawn as a placard at `LABEL_H`, sized by distance, so it reads as a sign
    by the door rather than a caption over the corridor.

    Drawn nearest first, and a name whose box would touch one already placed
    is dropped rather than drawn over it. Painting the far one underneath
    instead is what the first version did, and two labels a few pixels apart
    interleave into something that reads as neither -- "main entrance" and
    "library" came out as "mai library nce". Down a 48 m corridor seen nearly
    end-on, that overlap is the common case, not the edge one.

    Faded by distance as well, so the corridor ahead does not turn into a list.
    """
    from PIL import Image, ImageDraw

    near = []
    for name, (x, y) in places:
        d = math.dist(dog_xy, (x, y))
        if d > radius:
            continue
        px = cam.pixel((x, y, floor_z + LABEL_H))
        if px is None:
            continue
        u, v = px
        if not (0 <= u < cam.w and 0 <= v < cam.h):
            continue
        near.append((d, name, u, v))
    if not near:
        return img

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    placed = []
    for d, name, u, v in sorted(near):
        size = max(SIGN_MIN, min(SIGN_MAX, int(round(SIGN_PX / max(d, 0.5)))))
        font = _font(size)
        pad = max(3, size // 4)
        box = draw.textbbox((u, v), name, font=font, anchor="mm")
        box = (box[0] - pad, box[1] - pad + 1, box[2] + pad, box[3] + pad - 1)
        if any(box[0] <= q[2] and q[0] <= box[2] and
               box[1] <= q[3] and q[1] <= box[3] for q in placed):
            continue
        placed.append(box)
        # Full strength until the last couple of metres of range, then out.
        # The floor is high because a sign that dims into the wall is the
        # thing this is meant to stop being.
        fade = min(1.0, (radius - d) / FADE_R)
        fg = tuple(int(160 + (255 - 160) * fade) for _ in range(3))
        draw.rectangle(list(box), fill=PLATE, outline=EDGE, width=1)
        draw.text((u, v), name, font=font, fill=fg, anchor="mm")
    img[:] = np.asarray(im)
    return img
