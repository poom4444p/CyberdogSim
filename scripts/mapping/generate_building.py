"""Generate a synthetic multi-floor building that matches the location names
the language layer's outputs (cyberdog/language/generate_dataset.py).

PLACEHOLDER: replace with the real LumaAI scan + real room layout once
available. Floor/room counts are imported from generate_dataset.py so the
map and the NLU model can't drift apart.

Outputs (in data/building/):
    floor1.obj, floor2.obj, ...   Z-up meshes, same style as classroom_map.obj
    locations.json                name -> list of {floor, xy, door_xy}
    zones.json                    Behavior Layer polygons (the stairwells)

Then build the grids with scripts/mapping/mesh_to_grid.py --up-axis z --points 4000000
--inflate 0.25 (see src/cyberdog/planning/README.md).

Layout of every floor (top-down, meters):

    y=19 +------+------+-- ... --+------+
         | N0   | N1   |         | N7   |   north rooms (8 slots, 6 m wide)
    y=11 +--  --+--  --+-- ... --+--  --+   doorway gaps onto the corridor
         | S    corridor (3 m wide)     L |   S = stairs (west), L = lift (east)
    y=8  +--  --+--  --+-- ... --+--  --+
         | S0   | S1   |         | S7   |   south rooms
    y=0  +------+------+-- ... --+------+
        x=0                            x=48
"""
import json
import os

from cyberdog.language.generate_dataset import NUM_FLOORS, ROOMS_PER_FLOOR  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "building")

SLOTS_PER_SIDE = 8
ROOM_W = 6.0
ROOM_D = 8.0
CORRIDOR_W = 3.0
WALL_T = 0.2
WALL_H = 3.0
DOOR_W = 1.2

BUILDING_L = SLOTS_PER_SIDE * ROOM_W                  # 48 m
CORR_Y0 = ROOM_D                                      # 8
CORR_Y1 = ROOM_D + CORRIDOR_W                         # 11
BUILDING_W = 2 * ROOM_D + CORRIDOR_W                  # 19
CORR_MID_Y = (CORR_Y0 + CORR_Y1) / 2

STAIRS_XY = (1.0, CORR_MID_Y)
# The lift sits at the opposite, east dead-end of the corridor. Stairs are a
# hazard this robot refuses (blind user, no hand on a rail), so every floor
# change goes through here -- putting the two at opposite ends means a route
# that heads the wrong way is obvious on sight, in the map and in the video.
ELEVATOR_XY = (BUILDING_L - 1.4, CORR_MID_Y)

# The stairwell, as the Behavior Layer sees it: a STOP zone, not a route. It
# spans the full corridor width -- a no-go zone with a walkable channel down
# the middle would just be an invitation to thread between two flights -- and
# it stops at x=2.5, half a metre short of the doors of the westernmost rooms
# (slot 0, gaps centred on x=3), so sealing the stairwell does not also seal
# room 101 and room 106 off the map. sim/stairs.py's shaft is sized to match.
STAIRS_ZONE = {"x": (0.0, 2.5), "y": (8.3, 10.7)}

# Named (non-numbered) locations, pinned to fixed slots so the same place
# sits in the same spot on every floor it appears on. Names must match the
# canonical names in cyberdog/language/generate_dataset.py exactly.
# (side, slot): side "S"/"N" of the corridor, slot 0-7 from west to east.
# Numbered rooms take slots 0-4 on both sides; these use slots 5-7.
# PLACEHOLDER assignment until the real building layout is known.
#
#            slot 5            slot 6                      slot 7
#   N  F1: cafeteria         F1: -                       F1: library
#      F2: electrical eng    F2: mechanical eng          F2: library (2nd floor of it)
#      F3: biology lab       F3: physics lab             F3: -
#   S  all: restroom         F1: office                  F1: main entrance
#      (one per floor,       F2: computer eng lab        F2: -
#       stacked)             F3: chemistry lab           F3: server room
NAMED_ROOMS = {
    1: {("S", 5): "restroom", ("S", 6): "office", ("S", 7): "main entrance",
        ("N", 5): "cafeteria", ("N", 7): "library"},
    2: {("S", 5): "restroom", ("S", 6): "computer engineering lab",
        ("N", 5): "electrical engineering lab", ("N", 6): "mechanical engineering lab",
        ("N", 7): "library"},
    3: {("S", 5): "restroom", ("S", 6): "chemistry lab", ("S", 7): "server room",
        ("N", 5): "biology lab", ("N", 6): "physics lab"},
}


def write_box(f, x1, y1, z1, x2, y2, z2, v_idx):
    """Same triangulated box writer as generate_obj.py."""
    verts = [
        (x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1),
        (x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)
    ]
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for a, b, c in [(1, 4, 3), (1, 3, 2), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
                    (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]:
        f.write(f"f {v_idx+a} {v_idx+b} {v_idx+c}\n")
    return v_idx + 8


def slot_positions(side, slot):
    """Room center and the corridor point in front of its door."""
    x_mid = slot * ROOM_W + ROOM_W / 2
    y_room = ROOM_D / 2 if side == "S" else CORR_Y1 + ROOM_D / 2
    return (x_mid, y_room), (x_mid, CORR_MID_Y)


def floor_slots(floor):
    """(name, side, slot) for every labeled room on one floor."""
    half = (ROOMS_PER_FLOOR + 1) // 2
    used = {"S": half, "N": ROOMS_PER_FLOOR - half}  # numbered rooms take slots 0..used-1
    for (side, slot), name in NAMED_ROOMS.get(floor, {}).items():
        if slot < used[side] or slot >= SLOTS_PER_SIDE:
            raise ValueError(f"'{name}' on floor {floor} ({side}, {slot}) overlaps a numbered "
                             f"room or is off the building; adjust NAMED_ROOMS/SLOTS_PER_SIDE.")
    out = []
    for n in range(1, ROOMS_PER_FLOOR + 1):
        side, slot = ("S", n - 1) if n <= half else ("N", n - 1 - half)
        out.append((f"room {floor}{n:02d}", side, slot))
    out += [(name, side, slot) for (side, slot), name in NAMED_ROOMS.get(floor, {}).items()]
    return out


def write_floor_obj(path):
    t = WALL_T / 2
    with open(path, "w") as f:
        f.write("# Synthetic building floor (placeholder), Z-up, meters\n")
        v = 0
        # Floor slab
        v = write_box(f, -t, -t, -0.1, BUILDING_L + t, BUILDING_W + t, 0, v)
        # Outer walls
        v = write_box(f, -t, -t, 0, BUILDING_L + t, t, WALL_H, v)
        v = write_box(f, -t, BUILDING_W - t, 0, BUILDING_L + t, BUILDING_W + t, WALL_H, v)
        v = write_box(f, -t, -t, 0, t, BUILDING_W + t, WALL_H, v)
        v = write_box(f, BUILDING_L - t, -t, 0, BUILDING_L + t, BUILDING_W + t, WALL_H, v)
        # Partitions between rooms (both sides of the corridor)
        for k in range(1, SLOTS_PER_SIDE):
            x = k * ROOM_W
            v = write_box(f, x - t, 0, 0, x + t, CORR_Y0, WALL_H, v)
            v = write_box(f, x - t, CORR_Y1, 0, x + t, BUILDING_W, WALL_H, v)
        # Corridor walls with one doorway gap per room
        for y in (CORR_Y0, CORR_Y1):
            for k in range(SLOTS_PER_SIDE):
                x0 = k * ROOM_W
                gap0 = x0 + ROOM_W / 2 - DOOR_W / 2
                gap1 = x0 + ROOM_W / 2 + DOOR_W / 2
                v = write_box(f, x0, y - t, 0, gap0, y + t, WALL_H, v)
                v = write_box(f, gap1, y - t, 0, x0 + ROOM_W, y + t, WALL_H, v)


def write_zones(path):
    """Behavior Layer polygons for the building.

    Only the stairwells so far. They are written from the same constants the
    stairs are drawn from, so the zone cannot end up somewhere the flight
    isn't. Action rules are left empty on purpose -- BehaviorLayer.from_config
    fills them from config/map_config.yaml, which is where the
    "stairs = full stop, announce, requires confirmation" rule lives.

    The zone repeats per floor even though the geometry is identical, because
    the stairwell is a real hazard on each of them and the metadata is what
    tells you which floor a trigger fired on.
    """
    (zx0, zx1), (zy0, zy1) = STAIRS_ZONE["x"], STAIRS_ZONE["y"]
    zones = [{
        "zone_id": f"stairs_f{floor}",
        "zone_type": "stairs",
        "polygon": [[zx0, zy0], [zx1, zy0], [zx1, zy1], [zx0, zy1]],
        "metadata": {"name": f"Floor {floor} stairwell", "floor": floor},
    } for floor in range(1, NUM_FLOORS + 1)]

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"zones": zones}, f, indent=2)
    print(f"Wrote {path} ({len(zones)} zones)")


def main():
    locations = {}

    def add(name, entry):
        locations.setdefault(name, []).append(entry)

    for floor in range(1, NUM_FLOORS + 1):
        obj_path = os.path.join(OUT_DIR, f"floor{floor}.obj")
        write_floor_obj(obj_path)
        print(f"Wrote {obj_path}")

        for name, side, slot in floor_slots(floor):
            xy, door_xy = slot_positions(side, slot)
            add(name, {"floor": floor, "xy": list(xy), "door_xy": list(door_xy)})

        add("hallway", {"floor": floor, "xy": [BUILDING_L / 2, CORR_MID_Y],
                        "door_xy": [BUILDING_L / 2, CORR_MID_Y]})
        # Every word for a staircase, same point. BuildingRouter.hazard_named
        # matches the command against these names before the NLU model runs;
        # a synonym missing here goes to the model, which maps it onto the
        # nearest place it knows ("staircase" -> "hallway") without a word.
        for name in ("stairs", "staircase", "stairway", "stairwell"):
            add(name, {"floor": floor, "xy": list(STAIRS_XY), "door_xy": list(STAIRS_XY)})
        # Both names, same point: the destination lookup in cyberdog/sim/run_building.py
        # and the planning layer match on the literal word the user said.
        for name in ("elevator", "lift"):
            add(name, {"floor": floor, "xy": list(ELEVATOR_XY),
                       "door_xy": list(ELEVATOR_XY)})

    loc_path = os.path.join(OUT_DIR, "locations.json")
    with open(loc_path, "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=2)
    print(f"Wrote {loc_path} ({len(locations)} names)")

    write_zones(os.path.join(OUT_DIR, "zones.json"))


if __name__ == "__main__":
    main()
