"""Turn a floor's occupancy grid into a MuJoCo scene.

MuJoCo collides meshes as their convex hull, so dropping floor1.obj in would
make the whole building one solid block with no corridors. Instead we rebuild
the walls as boxes straight from the grid the planner already uses -- same
frame, same coordinates, and the doorways survive.

With one correction, and it is not cosmetic. The stored grids are already
inflated by the robot's radius (map_config grid_inflation, 0.25 m) so that A*
can treat the dog as a point. Building the scene from those directly draws
every wall 0.25 m into the room on each side: the corridor comes out 2.2 m
instead of its real 2.8 m, and a dog whose *centre* is legitimately on the
inflated boundary has half of itself inside the wall. It looks like a clipping
bug and it is really a category error -- the grid says where a point robot may
go, not where the building is.

Worse than the looks: the dog's camera and its LiDAR are supposed to be
measuring the world, and they were measuring the planner's opinion of it, a
corridor 0.6 m narrower than the one that exists. So the walls are eroded back
by the same inflation before they are drawn. The planning grids are untouched
-- they should be inflated, that is their job.
"""
import math
import os
import yaml
import numpy as np
from PIL import Image

from cyberdog.sim.scene import lift
from cyberdog.sim.scene import obstacles
from cyberdog.sim.scene import stairs
from cyberdog import paths
from cyberdog.paths import (BUILDING_MAPS as BUILDING, GO2_XML, MENAGERIE,
                            WALL_HEIGHT)
from cyberdog.sim.scene.levels import SLAB_THICK, floor_z

# How much the stored grids were inflated by, read from the same file the
# planner reads so the scene and the router cannot disagree about it.
INFLATION = yaml.safe_load(
    open(paths.MAP_CONFIG)
)["map"]["grid_inflation"]


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


def wall_grid(floor):
    """The occupancy grid with the planner's inflation taken back off.

    Erosion is not an exact inverse of dilation, but every wall here is far
    thicker than the 0.25 m structuring element, so it recovers them; doorway
    gaps widen back out by the same amount. `border_value=1` keeps the outer
    walls from eroding away against the edge of the image.
    """
    occ, res, ox, oy = load_grid(floor)
    cells = int(round(INFLATION / res))
    if cells > 0:
        from scipy import ndimage
        occ = ndimage.binary_erosion(
            occ, np.ones((2 * cells + 1, 2 * cells + 1), dtype=bool),
            border_value=1)
    return occ, res, ox, oy


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


def rects_to_geoms(rects, res, ox, oy, H, z0=0.0, prefix="wall"):
    """Grid rectangles -> MuJoCo box geoms, standing on the slab at z0."""
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

        geoms.append( f'<geom name="{prefix}{i}" type="box" '
              f'pos="{cx:.3f} {cy:.3f} {z0 + hz:.3f}" '
              f'size="{hx:.3f} {hy:.3f} {hz:.3f}" '
              f'rgba="0.78 0.78 0.83 1"/>')
    return geoms


def box(name, x0, x1, y0, y1, z0, z1, rgba):
    """A box from two corners -- MuJoCo wants centre and half-size."""
    return (f'<geom name="{name}" type="box" '
            f'pos="{(x0 + x1) / 2:.3f} {(y0 + y1) / 2:.3f} {(z0 + z1) / 2:.3f}" '
            f'size="{(x1 - x0) / 2:.3f} {(y1 - y0) / 2:.3f} {(z1 - z0) / 2:.3f}" '
            f'rgba="{rgba}"/>')


def subtract(rects, gap):
    """Every rect in `rects`, with `gap` cut out of it.

    A rect that overlaps the gap comes back as up to four -- below it, above
    it, and the strips left and right. Holes are punched one at a time, so a
    slab with a stairwell at one end and a lift shaft at the other is the same
    code as a slab with one opening.
    """
    if gap is None:
        return rects
    gx0, gx1, gy0, gy1 = gap
    out = []
    for x0, x1, y0, y1 in rects:
        if gx1 <= x0 or gx0 >= x1 or gy1 <= y0 or gy0 >= y1:
            out.append((x0, x1, y0, y1))          # misses this piece entirely
            continue
        cx0, cx1 = max(x0, gx0), min(x1, gx1)     # the overlap itself
        if y0 < gy0:
            out.append((x0, x1, y0, gy0))
        if gy1 < y1:
            out.append((x0, x1, gy1, y1))
        if x0 < cx0:
            out.append((x0, cx0, max(y0, gy0), min(y1, gy1)))
        if cx1 < x1:
            out.append((cx1, x1, max(y0, gy0), min(y1, gy1)))
    return out


def slab_geoms(n, extent):
    """Floor n's plate, with its openings cut out.

    Two of them from floor 2 up: the stairwell, which is what lets the dog's
    camera see that there are stairs there at all, and the lift shaft, which
    is what the car rises through.
    """
    X0, X1, Y0, Y1 = extent
    z = floor_z(n)
    zb, rgba = z - SLAB_THICK, "0.42 0.43 0.46 1"

    rects = [(X0, X1, Y0, Y1)]
    for gap in (stairs.hole(n), lift.hole(n)):
        rects = subtract(rects, gap)
    return [box(f"slab{n}_{i}", x0, x1, y0, y1, zb, z, rgba)
            for i, (x0, x1, y0, y1) in enumerate(rects)]


def stair_geoms(from_floor):
    """The flight from one floor up to the next -- scenery, and a hazard.

    Nothing climbs it (see stairs.py); it is here so the dog's camera and
    VAMOS's candidate paths have the real thing to be scored against.
    """
    z0 = floor_z(from_floor)
    return [box(f"step{from_floor}_{i}", x0, x1, y0, y1, z0 - SLAB_THICK, zt,
                "0.62 0.60 0.58 1")
            for i, (x0, x1, y0, y1, zt) in enumerate(stairs.steps(from_floor))]


def camera_params():
    """Vertical FOV in degrees and lens height, from the projector's config.

    Read from the same file so the sim camera and checkpoint_projector can't
    quietly disagree about what the robot sees.
    """
    p = paths.CAMERA_CONFIG
    c = yaml.safe_load(open(p))
    fovy = math.degrees(2 * math.atan((c["height"] / 2) / c["fy"]))
    return fovy, c["camera_height"]


def build(floor=1):
    """Full scene XML for one floor: the Go2, the walls and the onboard camera."""
    occ, res, ox, oy = wall_grid(floor)
    rects = find_rects(occ)
    geoms = rects_to_geoms(rects, res, ox, oy, occ.shape[0])
    # Not from the grid -- see obstacles.py. This scene builds every storey at
    # z=0, so the boxes stand on the ground rather than on their own floor.
    geoms += obstacles.geoms(floor, z0=0.0)

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

def build_building(floors=(1, 2, 3)):
    """The whole stack in one scene: every floor's walls and slab, the lift
    shaft that joins them, and the stairwells it refuses.

    One model instead of one per floor, so a route that crosses floors is a
    single continuous run -- the dog rides the lift in view instead of the
    video cutting to a new scene.
    """
    floors = sorted(floors)
    parts, lights = [], []
    extent = None

    for n in floors:
        occ, res, ox, oy = wall_grid(n)
        H, W = occ.shape
        extent = (ox, ox + W * res, oy, oy + H * res)
        z = floor_z(n)

        parts += slab_geoms(n, extent)
        parts += rects_to_geoms(find_rects(occ), res, ox, oy, H, z0=z, prefix=f"f{n}w")
        parts += obstacles.geoms(n)
        if n != floors[-1]:
            parts += stair_geoms(n)

        # Each slab is a ceiling for the floor below, so the sun never reaches
        # past floor 1. One light per storey, just under its own ceiling.
        cx, cy = (W * res) / 2, (H * res) / 2
        lights.append(f'<light name="floor{n}" pos="{cx} {cy} {z + WALL_HEIGHT - 0.1:.2f}" '
                      f'dir="-0.2 -0.15 -1" directional="true" '
                      f'diffuse="0.45 0.45 0.47" specular="0.05 0.05 0.05"/>')

    parts += lift.shaft_geoms(floors)

    X0, X1, Y0, Y1 = extent
    cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
    top = floor_z(floors[-1]) + WALL_HEIGHT

    # The car is a mocap body, not a jointed one: mocap bodies live outside
    # qpos, so the Go2's "home" keyframe still fits the model and the dog's
    # qpos[0:7] stay where MujocoRobot expects them. run_building.py moves it
    # with the dog.
    cx0, cy0, cz0 = lift.car_pos(floor_z(floors[0]))
    car = f"""<body name="lift_car" mocap="true" pos="{cx0:.3f} {cy0:.3f} {cz0:.3f}">
      {chr(10).join('  ' + g for g in lift.car_geoms())}
    </body>"""
    fovy, cam_h = camera_params()
    body = "\n    ".join(lights + parts)

    return f"""<mujoco model="cyberdog_building">
  <include file="{GO2_XML}"/>
  <compiler meshdir="{MENAGERIE}/unitree_go2/assets"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.42 0.42 0.45" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <rgba haze="0.6 0.62 0.68 1"/>
    <map znear="0.002" zfar="60"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.62 0.72"
             rgb2="0.82 0.85 0.9" width="512" height="512"/>
  </asset>

  <worldbody>
    <light name="sun" pos="{cx} {cy} {top + 6:.2f}" dir="-0.35 -0.25 -1" directional="true"
           diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <geom name="ground" type="plane" pos="{cx} {cy} {-SLAB_THICK - 0.01:.3f}"
          size="{(X1 - X0) / 2 + 6:.2f} {(Y1 - Y0) / 2 + 6:.2f} 0.1" rgba="0.34 0.36 0.38 1"/>
    {body}
    {car}
    <camera name="dogcam" pos="0 0 {cam_h}" fovy="{fovy:.2f}"/>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    out_dir = paths.ensure_output() and paths.SCENE_CACHE

    if args and args[0] == "--building":
        floors = [int(a) for a in args[1:]] or [1, 2, 3]
        xml = build_building(floors)
        out = os.path.join(out_dir, "building.xml")
    else:
        floor = int(args[0]) if args else 1
        xml = build(floor)
        out = os.path.join(out_dir, f"floor{floor}.xml")

    with open(out, "w") as f:
        f.write(xml)
    print(f"wrote {out}  ({xml.count('<geom')} geoms)")
