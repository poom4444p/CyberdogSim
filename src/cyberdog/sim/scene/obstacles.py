"""Things in the corridor that the map does not know about.

Every other solid in this scene -- the walls, the slabs, the lift, the
stairwells -- is also in the occupancy grid the planner runs A* on. That makes
the whole perception half of the stack decorative: VAMOS proposes paths from a
picture, and those paths are then scored against the same ground truth the
route already came from, so the camera can never contribute anything the map
did not already have.

These boxes are the exception, and the exception is the point. They are drawn
into the scene and are **deliberately absent from data/building** --
no grid cell, no zone, no location. The map is stale on purpose. A* will route
straight through them, clearance_test() will report open corridor where a
crate is standing, and dreaming.py will happily imagine a rollout passing
through one. That is not a bug to fix here: it is the gap that gives the
perception layer a job, and the first run that walks the dog into one of these
is the experiment.

Nothing in this module detects anything. Closing the gap -- folding what the
camera can actually see into the clearance field -- is the next step and is
deliberately not taken yet (see README, "Known limits").

Placement. The walkable corridor after the grid's 0.25 m inflation is
y 8.425 - 10.575, so 2.15 m wide. Each box takes a bite out of one side and
leaves the other clear -- an obstruction rather than a wall.

Which side is not a matter of taste, and there are two lines to miss. A*'s
checkpoints hug whichever wall the destination is on -- about y=8.6 going to a
south room, y=10.3 going to a north one -- but the dog does not drive the
polyline. Pure-pursuit cuts every corner, so the long transits are actually
walked down the middle at y=9.5 (run any cross-corridor leg with --no-video and
read the printed positions). A box that clears both lines is geometrically in
the corridor and never once in the dog's way: a very reassuring video and no
experiment at all.

So each box spans from its own side across the centreline, and CENTRELINE below
is asserted, not hoped for. Placement along x follows the routes that actually
cross that stretch:

    floor 1  south, at the midpoint -- the long legs to the west rooms run
             down the south wall as far as x=17 whichever side they end on
    floor 2  a pair 18 m apart on the westward transit
    floor 3  a few metres out of the lift, so the dog turns out of the car and
             meets it rather than watching it approach for forty metres

Re-check that after any change to the grids or the router -- a box the routes
have quietly stopped touching is worse than no box, because it still looks
like a test.
"""
from cyberdog.sim.scene.levels import floor_z

# Warm, so it reads as "not part of the building" at a glance in the video --
# every wall, slab and shaft in the scene is some shade of grey.
COLOUR = "0.80 0.45 0.22 1"

# The corridor, as the *inflated* grid sees it. Boxes are checked against this
# rather than against the raw 8 - 11 m walls, because a box in the inflated
# margin would be inside a wall as far as any route is concerned.
CORRIDOR_Y = (8.425, 10.575)
# Where the corridor walls actually are, now that build_scene draws them at
# their real positions rather than at the planner's inflated ones. A box that
# stops at the walkable band's edge leaves a visible slot to the wall that the
# planner will never route through, so the video shows a dog ignoring a gap it
# could obviously fit down. Boxes run to the wall; only the overlap with
# CORRIDOR_Y is what actually blocks anything.
WALL_Y = (8.15, 10.85)
CENTRELINE = 9.5        # where pure-pursuit actually walks the long transits
MIN_GAP = 0.60          # leave at least this much to squeeze past, metres

# (x0, x1, y0, y1, height, what it is). Heights are all well above the 0.32 m
# lens, so every one of them is something the camera plainly sees.
OBSTACLES = {
    1: [(23.40, 24.60, 8.15, 9.60, 0.85, "trolley")],
    2: [(34.00, 35.20, 9.45, 10.85, 0.70, "crate"),
        (16.00, 17.20, 9.40, 10.85, 0.95, "cart")],
    3: [(41.00, 42.20, 9.40, 10.85, 0.60, "cartons")],
}

# A box that sealed the corridor would be a wall the planner cannot see, and
# every route on that floor would fail for reasons no error message explains.
for _floor, _items in OBSTACLES.items():
    for _x0, _x1, _y0, _y1, _h, _name in _items:
        assert WALL_Y[0] <= _y0 < _y1 <= WALL_Y[1], \
            f"{_name} on floor {_floor} is outside the corridor walls"
        assert _y0 < CORRIDOR_Y[1] and _y1 > CORRIDOR_Y[0], \
            f"{_name} on floor {_floor} blocks nothing the planner can use"
        # Only the part inside the walkable band obstructs anything.
        _gap = max(min(_y0, CORRIDOR_Y[1]) - CORRIDOR_Y[0],
                   CORRIDOR_Y[1] - max(_y1, CORRIDOR_Y[0]))
        assert _gap >= MIN_GAP, \
            f"{_name} on floor {_floor} leaves only {_gap:.2f} m to pass"
        # The one that actually decides whether the dog meets it.
        assert _y0 <= CENTRELINE <= _y1, \
            (f"{_name} on floor {_floor} does not reach the centreline the dog "
             f"walks -- it would sit in the corridor and never be in the way")


def boxes(floor):
    """(x0, x1, y0, y1, height, name) for everything standing on this floor."""
    return OBSTACLES.get(floor, [])


def geoms(floor, z0=None):
    """MuJoCo box geoms for floor `floor`, standing on its walking surface.

    `z0` overrides the surface height, for the single-floor scenes that build
    every storey at z=0.
    """
    z = floor_z(floor) if z0 is None else z0
    out = []
    for x0, x1, y0, y1, h, name in boxes(floor):
        out.append(
            f'<geom name="obs{floor}_{name}" type="box" '
            f'pos="{(x0 + x1) / 2:.3f} {(y0 + y1) / 2:.3f} {z + h / 2:.3f}" '
            f'size="{(x1 - x0) / 2:.3f} {(y1 - y0) / 2:.3f} {h / 2:.3f}" '
            f'rgba="{COLOUR}"/>')
    return out
