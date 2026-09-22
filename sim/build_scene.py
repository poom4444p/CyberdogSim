"""Turn a floor's occupancy grid into a MuJoCo scene.

MuJoCo collides meshes as their convex hull, so dropping floor1.obj in would
make the whole building one solid block with no corridors. Instead we rebuild
the walls as boxes straight from the grid the planner already uses -- same
frame, same coordinates, and the doorways survive.
"""
import math
import os
import yaml
import numpy as np
from PIL import Image

from config import BUILDING, GO2_XML, MENAGERIE, ROOT, WALL_HEIGHT


def load_grid(floor):
    """Read one floor's grid. Row 0 is the top of the map -- highest world y."""
    d = os.path.join(BUILDING, f"floor{floor}")

    with open(os.path.join(d, "room_map.yaml")) as f:
        meta = yaml.safe_load(f)

    res = meta["resolution"]
    ox, oy = meta["origin"][0], meta["origin"][1]

    img = np.array(Image.open(os.path.join(d, "room_map.png")))
    if img.ndim == 3:
        img = img[..., 0]

    occupied = img <= 100   # 0 is wall, 254 is free space
    return occupied, res, ox, oy


def find_rects(occupied):
    """Merge wall cells into rectangles -- 83k cells becomes ~166 boxes.

    Returns (row0, row1, col0, col1), all inclusive.
    """
    H, W = occupied.shape
    rects = []
    open_rects = {}

    for row in range(H):
        # Runs of wall cells in this row. Looping to W+1 lets the imaginary
        # last column close the final run instead of a special case after.
        runs = []
        start = None
        for col in range(W + 1):
            filled = col < W and occupied[row, col]
            if filled and start is None:
                start = col
            elif not filled and start is not None:
                runs.append((start, col - 1))
                start = None

        seen = set()
        for col in runs:
            seen.add(col)
            # Same span directly above -> grow it downwards. The row-1 check
            # matters: without it a wall above a doorway merges with the wall
            # below it and seals the door shut.
            if col in open_rects and open_rects[col][1] == row -1:
                open_rects[col][1] = row
            else:
                if col in open_rects:
                    r0, r1 = open_rects[col]
                    rects.append((r0, r1, col[0], col[1]))
                open_rects[col] = [row, row]

        # Anything that didn't continue into this row is finished.
        for col in [k for k in open_rects if k not in seen]:
            r0, r1 = open_rects.pop(col)
            rects.append((r0, r1, col[0], col[1]))

    for col, (r0, r1) in open_rects.items():
        rects.append((r0, r1, col[0], col[1]))

    return rects


def rects_to_geoms(rects, res, ox, oy, H):
    """Grid rectangles -> MuJoCo box geoms."""
    geoms = []
    hz = WALL_HEIGHT / 2.0

    for i, (row0, row1, col0, col1) in enumerate(rects):
        # +1 spans the full last cell, otherwise every wall comes out one
        # cell thin. row1 is the bottom row so it gives the lower y.
        x0 = ox + col0 * res
        x1 = ox + (col1 + 1) * res
        y0 = oy + (H - 1 - row1) * res
        y1 = oy + (H - 1 - row0 + 1) * res

        # MuJoCo wants centre and half-size, not corners.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx, hy = (x1 - x0) / 2, (y1 - y0) / 2

        geoms.append( f'<geom name="wall{i}" type="box" '
              f'pos="{cx:.3f} {cy:.3f} {hz:.3f}" '
              f'size="{hx:.3f} {hy:.3f} {hz:.3f}" '
              f'rgba="0.78 0.78 0.83 1"/>')
    return geoms


def camera_params():
    """Vertical FOV in degrees and lens height, from the projector's config.

    Read from the same file so the sim camera and checkpoint_projector can't
    quietly disagree about what the robot sees.
    """
    p = os.path.join(ROOT, "planner_layer", "camera_config.yaml")
    c = yaml.safe_load(open(p))
    fovy = math.degrees(2 * math.atan((c["height"] / 2) / c["fy"]))
    return fovy, c["camera_height"]


def build(floor=1):
    """Full scene XML for one floor: the Go2, the walls and the onboard camera."""
    occ, res, ox, oy = load_grid(floor)
    rects = find_rects(occ)
    geoms = rects_to_geoms(rects, res, ox, oy, occ.shape[0])

    H, W = occ.shape
    cx, cy = (W * res) / 2, (H * res) / 2

    walls = "\n ".join(geoms)
    fovy, cam_h = camera_params()
    # The compiler line has to come after the include: go2.xml carries its own
    # with a relative meshdir, and whichever compiler is last wins.
    return f"""<mujoco model="cyberdog_floor{floor}">
  <include file="{GO2_XML}"/>
  <compiler meshdir="{MENAGERIE}/unitree_go2/assets"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <!-- Walls are vertical, so a straight-down light leaves every face black.
         Ambient headlight lifts them; the slanted lights give edges definition. -->
    <headlight ambient="0.42 0.42 0.45" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <rgba haze="0.6 0.62 0.68 1"/>
    <!-- znear is a fraction of model extent; the building is 48 m across, so
         the default clips the floor right in front of a 0.32 m lens. -->
    <map znear="0.002" zfar="30"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.62 0.72"
             rgb2="0.82 0.85 0.9" width="512" height="512"/>
  </asset>

  <worldbody>
    <light name="sun" pos="{cx} {cy} 8" dir="-0.35 -0.25 -1" directional="true"
           diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <light name="fill" pos="{cx} {cy} 6" dir="0.4 0.3 -1" directional="true"
           diffuse="0.3 0.3 0.34" specular="0 0 0"/>
    <geom name="ground" type="plane" pos="{cx} {cy} 0"
          size="{cx + 5} {cy + 5} 0.1" rgba="0.42 0.43 0.46 1"/>
      {walls}
    <camera name="dogcam" pos="0 0 {cam_h}" fovy="{fovy:.2f}"/>
  </worldbody>
</mujoco>
"""

if __name__ == "__main__":
    import sys
    floor = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    xml = build(floor)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "scene_cache", f"floor{floor}.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"wrote {out}  ({xml.count('<geom')} geoms)")
