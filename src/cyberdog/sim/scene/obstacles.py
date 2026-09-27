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

Size. They are the objects they are named after and nothing bigger: a flatbed
trolley 1.2 m by 0.7 m, a janitor's cart 0.9 by 0.55, a pallet crate 1.0 by
0.8, a stack of cartons 0.6 square. That is the whole point of them -- a thing
somebody left in a corridor, which is what the dog will actually meet. An
earlier version ran each box from a wall across the centreline, 1.45 m deep in
a 2.7 m corridor, and the size was doing the work the perception layer is
supposed to do: over half the hallway is not an obstacle to be seen and gone
round, it is a chicane, and getting past it is a full-corridor crossing that
needs a planned curve nothing here has yet. That is what the `room 101` route
used to stop in front of, and it now walks past. The old size also did not look
like anything -- no trolley is that deep, and a video of the dog halting in
front of one proves nothing about corridors that have trolleys in them.

Placement. The walkable corridor after the grid's 0.25 m inflation is
y 8.425 - 10.575, so 2.15 m wide. A realistic footprint leaves room on both
sides of it, so what decides whether the dog ever meets the thing is not its
size but where it sits across the corridor.

There is no single line to aim at, which is the difficulty. A*'s checkpoints hug
whichever wall the destination is on -- about y=8.6 going to a south room,
y=10.3 going to a north one -- but the dog does not drive the polyline.
Pure-pursuit cuts every corner off it, and where that lands depends on the leg:
measured with --no-video, the floor-2 westward transit runs at y 8.9 - 9.5, and
the floor-1 leg to the west rooms at y 8.3 - 8.5. A box parked tidily against a
wall misses every one of those lines: it is geometrically in the corridor and
never once in the dog's way -- a very reassuring video and no experiment at all.

So every box straddles the centreline, which is what CENTRELINE below asserts,
and the slack goes to one side rather than into the object. That puts it within
a metre of every line above, which is what actually matters: the dog does not
have to be aimed at the thing to have to deal with it.

Which side takes the slack still matters, and the reason is worth recording
because it was misdiagnosed once. Each box puts its slack into one ~1 m gap on
the side the routes through that stretch use -- all four open north, because
that is the side the doorways these routes turn into are on.

That layout was arrived at while `free_destination` had three bugs in it
(avoidance switched off whenever the goal pixel left the camera frame; straight
lines drawn through walls to destinations around corners; the sideways crossing
finished at the destination rather than at the obstacle). Those made the dog
look as though it needed a metre of corridor to commit to anything, and the
boxes were laid out around that apparent appetite. With them fixed it passes
what it fits through, and the layout is no longer load-bearing -- it is simply
where these objects sit. Someone rearranging them should re-run the twenty
routes rather than trusting the "open north" rule.

The floor-3 compromise is the tightest. Those routes fan out to doors on
*both* sides of the corridor from a single lift, so whichever side the cartons
leave open, routes wanting the other side get 0.52 m. `chemistry lab` is the
one that wants it: it used to drive through the cartons, then stopped short of
its door, and now gets in with 0.32 m in hand. Getting in means passing them on
the north and turning 90 degrees south into a door 1.2 m further on, which is a
curve -- CE-RRT*, spec L6 s2 -- and not something a goal displaced sideways can
express. What it turned out to need first was simply not changing its mind
about which side to pass on: see `free_destination(prefer=...)`.

MIN_GAP is asserted against the wider side, so it cannot catch a box whose open
side is the wrong one: which side is useful depends on the routes, not on the
geometry. Re-check it after any change to the grids or the router, along with
the two failures it cannot see -- a box the routes have quietly stopped touching
is worse than no box, because it still looks like a test, and a box that seals
the side a doorway is on breaks that route without ever being mentioned.
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
# their real positions rather than at the planner's inflated ones. Nothing here
# is allowed outside them -- a box half inside a wall is a modelling mistake
# that reads on camera as a smaller box.
WALL_Y = (8.15, 10.85)
CENTRELINE = 9.5        # where pure-pursuit actually walks the long transits
MIN_GAP = 0.60          # leave at least this much to squeeze past, metres

# (x0, x1, y0, y1, height, what it is). Footprints are the real objects':
# a flatbed trolley is 1.2 m by 0.7 m, not 1.2 by 1.45. Heights are all well
# above the 0.32 m lens, so every one of them is something the camera plainly
# sees.
OBSTACLES = {
    1: [(23.40, 24.60, 9.05, 9.75, 0.95, "trolley")],
    2: [(34.10, 35.10, 8.75, 9.55, 0.70, "crate"),
        (16.00, 16.90, 9.00, 9.55, 0.95, "cart")],
    3: [(41.20, 41.80, 8.95, 9.55, 0.90, "cartons")],
}

# A box that sealed the corridor would be a wall the planner cannot see, and
# every route on that floor would fail for reasons no error message explains.
for _floor, _items in OBSTACLES.items():
    for _x0, _x1, _y0, _y1, _h, _name in _items:
        assert WALL_Y[0] <= _y0 < _y1 <= WALL_Y[1], \
            f"{_name} on floor {_floor} is outside the corridor walls"
        assert _y0 < CORRIDOR_Y[1] and _y1 > CORRIDOR_Y[0], \
            f"{_name} on floor {_floor} blocks nothing the planner can use"
        # Both sides have to be a way past, not just the wider one. A box that
        # pins one side shut is fine until a route wants that side -- and then
        # it does not fail as an obstacle, it fails as a corridor: the dog
        # crawls up to a doorway it cannot reach and ends the leg. The floor-2
        # crate did exactly that to `electrical engineering lab`, whose door is
        # a metre west of it on the north wall, for as long as the crate ran to
        # the north wall. Only the part inside the walkable band obstructs
        # anything, so both sides are measured against that.
        _south = min(_y0, CORRIDOR_Y[1]) - CORRIDOR_Y[0]
        _north = CORRIDOR_Y[1] - max(_y1, CORRIDOR_Y[0])
        assert max(_south, _north) >= MIN_GAP, \
            (f"{_name} on floor {_floor} leaves {_south:.2f} m south and "
             f"{_north:.2f} m north -- nothing gets past it")
        # The one that actually decides whether the dog meets it.
        assert _y0 <= CENTRELINE <= _y1, \
            (f"{_name} on floor {_floor} does not straddle the centreline the "
             f"dog walks -- it would sit in the corridor and never be in the way")


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
