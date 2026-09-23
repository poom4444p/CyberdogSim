"""Where the stairwells are -- and they are scenery, not a route.

This robot leads someone who cannot see the steps and has no free hand for a
rail, so it never climbs. The flights are drawn anyway, and drawn where the
map says they are, because a hazard has to exist to be refused: the dog's
camera sees them, VAMOS's candidate paths get scored against them, and the
Behavior Layer's stop zone sits on top of them. Floor changes go through
lift.py.

build_scene.py draws the flights and the openings they come through. Nothing
walks them.

Layout. Every floor has "stairs" at the same spot in locations.json, the west
end of the corridor, so the shaft runs east from there along the corridor. The
corridor band is only 2.15 m wide (y 8.4 - 10.55), so the two flights share it
lengthwise: 1->2 uses the south lane, 2->3 the north one. That alternation is
what keeps a floor slab from having to be a hole and a stair base in the same
place -- floor 2's hole is in the south lane, and the 2->3 flight starts from
the north lane, which on floor 2 is still solid.
"""
from cyberdog.sim.scene.levels import FLOOR_HEIGHT, SLAB_THICK, floor_z  # noqa: F401  (re-exported)

# The flight's run, west to east. It has to end clear of the doors of the
# westernmost rooms (centred on x=3): the whole corridor width here is a no-go
# zone, and a zone that reached those doors would take rooms 101 and 106 off
# the map. Short and steep is fine -- nothing climbs this. See data/
# scripts/generate_building.py STAIRS_ZONE, which is the same numbers.
SHAFT_X0, SHAFT_X1 = 0.35, 2.2
N_STEPS = 14

# (y0, y1) of each lane, inside the corridor band and clear of y=9.5.
LANES = {"south": (8.45, 9.25), "north": (9.75, 10.55)}



def lane(from_floor):
    """Which lane the flight from `from_floor` up to the next one uses."""
    return "south" if from_floor % 2 else "north"


def hole(n):
    """(x0, x1, y0, y1) opening in floor n's slab, or None if it has none.

    Floor n is reached by the flight from n-1, so its hole is that flight's lane.
    """
    if n <= 1:
        return None
    y0, y1 = LANES[lane(n - 1)]
    return (SHAFT_X0 - 0.15, SHAFT_X1, y0, y1)


def steps(from_floor):
    """The flight from `from_floor` up one, as (x0, x1, y0, y1, z_top) boxes.

    Each step is solid down to the floor it starts from -- cheaper than treads
    and risers, and from the dog's camera it reads the same.
    """
    z0 = floor_z(from_floor)
    y0, y1 = LANES[lane(from_floor)]
    run = (SHAFT_X1 - SHAFT_X0) / N_STEPS
    rise = FLOOR_HEIGHT / N_STEPS
    return [(SHAFT_X0 + i * run, SHAFT_X0 + (i + 1) * run, y0, y1,
             z0 + (i + 1) * rise) for i in range(N_STEPS)]
