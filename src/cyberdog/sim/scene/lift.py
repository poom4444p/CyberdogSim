"""The lift at the east end of the corridor -- how this robot changes floors.

Stairs are a hazard it refuses (see stairs.py and the Behavior Layer's stop
zone); everything vertical happens here. build_scene.py draws the shaft and
the car, run_building.py rides them, and both read this module so the car
cannot stop at a height the scene has no opening at.

Layout. The shaft is the east dead-end of the corridor, hard against the
building's east wall, with its west face open as the doorway. Its footprint is
cut out of every upper slab, so the car -- and the dog standing in it -- passes
from storey to storey instead of into a ceiling. The stairwell is at the
opposite, west end: a route that heads the wrong way is wrong on sight.
"""
import json
import os

from cyberdog.paths import BUILDING_DIR
from cyberdog.sim.scene.levels import SLAB_THICK, floor_z

# Car footprint (x0, x1, y0, y1). Inside the corridor band (walls at y=8 and
# y=11) with room to spare around a 0.25 m robot radius, and running east to
# the outer wall, which doubles as the back of the shaft.
CAR = (45.6, 47.9, 8.6, 10.4)
CAR_H = 2.0                     # inside height, matching the corridor walls
WALL_T = 0.1
PLATE_T = 0.06                  # the car floor the dog stands on

RIDE_V = 0.6                    # m/s vertical -- slower than corridor pace

# Where the dog waits and rides: "elevator" in locations.json, the same point
# on every floor. Read from the map rather than repeated here, because a car
# drawn in one place and boarded from another is a dog stepping into a shaft.
with open(os.path.join(BUILDING_DIR, "locations.json"), encoding="utf-8") as _f:
    _entries = json.load(_f)["elevator"]
LIFT_XY = tuple(_entries[0]["xy"])
FLOORS_SERVED = sorted(e["floor"] for e in _entries)

_x0, _x1, _y0, _y1 = CAR
assert _x0 < LIFT_XY[0] < _x1 and _y0 < LIFT_XY[1] < _y1, \
    f"the elevator point {LIFT_XY} in locations.json is outside the car {CAR}"


def hole(n):
    """(x0, x1, y0, y1) opening in floor n's slab, or None if it has none.

    The car's own footprint: floor 1 stands on the ground, every floor above
    it has to let the car through.
    """
    return None if n <= 1 else CAR


def shaft_geoms(floors):
    """The three fixed walls of the shaft, full height of the stack.

    North, south and a west return on each side of the doorway; the east face
    is the building's own outer wall. The doorway is left open rather than
    given doors -- the dog is kinematic and would walk through them anyway,
    and an open face is what lets the chase camera watch the car rise.
    """
    x0, x1, y0, y1 = CAR
    zb = floor_z(min(floors)) - SLAB_THICK
    zt = floor_z(max(floors)) + CAR_H
    rgba = "0.38 0.40 0.45 1"
    return [
        _box("lift_s", x0 - WALL_T, x1, y0 - WALL_T, y0, zb, zt, rgba),
        _box("lift_n", x0 - WALL_T, x1, y1, y1 + WALL_T, zb, zt, rgba),
    ]


def car_geoms():
    """The car itself, in body-local coordinates (the body is the mocap).

    A floor plate and a lintel over the doorway: enough for the video to read
    as a car rising inside a shaft. It carries no collision role -- the dog is
    placed, not simulated -- so it is geometry only.
    """
    x0, x1, y0, y1 = CAR
    w, d = (x1 - x0) / 2, (y1 - y0) / 2
    return [
        f'<geom name="lift_plate" type="box" pos="0 0 {-PLATE_T / 2:.3f}" '
        f'size="{w:.3f} {d:.3f} {PLATE_T / 2:.3f}" rgba="0.55 0.56 0.60 1"/>',
        f'<geom name="lift_roof" type="box" pos="0 0 {CAR_H:.3f}" '
        f'size="{w:.3f} {d:.3f} 0.04" rgba="0.48 0.49 0.53 1"/>',
    ]


def car_pos(z):
    """World position of the car body when its floor is at height z."""
    x0, x1, y0, y1 = CAR
    return ((x0 + x1) / 2, (y0 + y1) / 2, z)


def _box(name, x0, x1, y0, y1, z0, z1, rgba):
    return (f'<geom name="{name}" type="box" '
            f'pos="{(x0 + x1) / 2:.3f} {(y0 + y1) / 2:.3f} {(z0 + z1) / 2:.3f}" '
            f'size="{(x1 - x0) / 2:.3f} {(y1 - y0) / 2:.3f} {(z1 - z0) / 2:.3f}" '
            f'rgba="{rgba}"/>')
