"""Multi-floor routing on top of the 2D A* planner.

Each floor has its own occupancy grid (data/building/maps/floorN/room_map.*).
Same floor: one A* call. Different floor: current floor -> lift, then
lift -> target on the target floor, with a "take the lift" announcement on the
handoff checkpoint.

The lift, not the stairs. This robot leads someone who cannot see the steps and
has no free hand for a rail, so a staircase is a hazard to be announced and
refused, never a way up: the stairwells are STOP zones in the Behavior Layer,
they are stamped into the planning grids as obstacles so A* cannot route
through one even by accident, and a floor the lift does not serve is
unreachable -- NoAccessibleRoute, with a reason to say out loud.

Running on Linux:
    Library module (imported by main_planner.py and test_locations.py), no GPU.
    Setup (once), from the project root:
        python3 -m venv .venv && source .venv/bin/activate
        pip install numpy pyyaml pillow scipy
    Maps must already exist in data/building/maps/ (they're generated
    files; rebuild with scripts/mapping/generate_building.py + mesh_to_grid.py,
    which additionally need `pip install open3d opencv-python`).
"""
import json
import math
import os
import re
from collections import deque

import yaml


from cyberdog.mapping.occupancy_grid import OccupancyGrid  # noqa: E402
from cyberdog.mapping.behavior_layer import BehaviorLayer  # noqa: E402
from cyberdog.mapping.astar_planner import AStarPlanner  # noqa: E402
from cyberdog.mapping.lane import (bridge_doors, door_gates, doorway_cells,  # noqa: E402
                                   keep_side, round_corners)

from cyberdog.paths import BUILDING_DIR, MAP_CONFIG as CONFIG_PATH

# The place every floor change goes through. "elevator" and "lift" are the same
# point in locations.json; this is the one the router asks for.
TRANSIT_NODE = "elevator"

# How far outside a no-go zone the dog stands when someone asks for a place
# inside one. Far enough that the run-time stop is not tripped by drift, close
# enough that "by the stairs" is an honest description of where you are.
APPROACH_CLEARANCE = 1.0


class NoAccessibleRoute(Exception):
    """The goal is only reachable a way this robot must refuse.

    Carries the sentence to say to the user -- callers print it instead of a
    bare "no route", because "there are only stairs" and "I could not find a
    path" are different things to be told when you cannot see.
    """


class BuildingRouter:
    def __init__(self, building_dir=BUILDING_DIR, config_path=CONFIG_PATH):
        with open(os.path.join(building_dir, "locations.json"), encoding="utf-8") as f:
            self.locations = json.load(f)
        self.floors = sorted({e["floor"] for entries in self.locations.values() for e in entries})
        self.grids = {
            n: OccupancyGrid.load(os.path.join(building_dir, "maps", f"floor{n}", "room_map"))
            for n in self.floors
        }
        # Plan on inflated copies so routes leave room for the robot's width;
        # self.grids stays raw for drawing and for test_locations.py. The
        # stored grids already carry grid_inflation (mesh_to_grid --inflate),
        # so only the remainder is added here -- inflating by the full radius
        # again narrowed every doorway to a couple of cells.
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["map"]
        radius = max(cfg.get("robot_radius", 0.0) - cfg.get("grid_inflation", 0.0), 0.0)
        self.plan_grids = {n: g.inflate(radius) for n, g in self.grids.items()}
        # Which side of a corridor to walk, and how far from its wall. The
        # config gives the distance from the wall face; the planning grids
        # already stop that much short of it (their total inflation), so the
        # lane is placed in their terms. See mapping/lane.py.
        lane = cfg.get("lane", {})
        self.lane_side = lane.get("side", "none")
        inflated = max(cfg.get("robot_radius", 0.0), cfg.get("grid_inflation", 0.0))
        self.lane_distance = max(lane.get("wall_distance", 0.0) - inflated, 0.0)
        self.lane_width = max(lane.get("max_corridor", 3.5) - 2 * inflated, 0.0)
        self.turn_radius = lane.get("turn_radius", 0.0)
        self._lane_walls = {}
        self.behavior = BehaviorLayer.from_config(
            config_path, zones_path=os.path.join(building_dir, "zones.json"))
        # A stop zone is not something to route through and halt at -- it is
        # something no route should contain. Stamping it into the planning
        # grids (not self.grids, which stays raw for drawing and tests) is what
        # makes that true for every A* call, on every floor.
        for g in self.plan_grids.values():
            self._block_zones(g)
        self.planner = AStarPlanner.from_config(config_path)

    def _block_zones(self, grid):
        """Mark every cell inside a full-stop zone as occupied."""
        for zone in self.behavior.zones:
            if zone.action_rules.speed_modifier > 0.0:
                continue
            xs = [p[0] for p in zone.polygon]
            ys = [p[1] for p in zone.polygon]
            r0, c0 = grid.world_to_grid(min(xs), min(ys))
            r1, c1 = grid.world_to_grid(max(xs), max(ys))
            for r in range(max(r0, 0), min(r1 + 1, grid.grid.shape[0])):
                for c in range(max(c0, 0), min(c1 + 1, grid.grid.shape[1])):
                    x, y = grid.grid_to_world(r, c)
                    if zone.contains_point(x, y):
                        grid.grid[r, c] = 100

    def hazard_named(self, text):
        """The no-go place named in a raw command, as (name, announcement).

        The intent parser is advisory and cannot be trusted with a safety
        rule: it was fine-tuned on destinations, so an unknown word comes back
        as the nearest one it does know -- "take me to the stairs" parses as
        "hallway", and the robot would set off for the middle of the corridor
        without ever mentioning the stairs. So the command is checked against
        the map's own names first, on word boundaries ("upstairs" is a floor
        phrase, not a request for a staircase).

        AI proposes, simple code disposes.
        """
        low = (text or "").lower()
        for name, entries in self.locations.items():
            if not re.search(rf"\b{re.escape(name)}\b", low):
                continue
            hazard = self.hazard_place(name)
            if hazard is not None:
                return name, hazard
        return None

    def hazard_place(self, name):
        """The announcement if `name` is a place inside a no-go zone, else None."""
        entries = self.locations.get((name or "").strip().lower(), [])
        return next((h for h in (self.hazard_at(e["xy"]) for e in entries) if h), None)

    def approach(self, name, current_floor, current_xy, floor=None):
        """Where to stand when someone asks for a place we will not enter.

        Asking for the stairs is a reasonable thing to ask: the person may be
        meeting someone there, or getting their bearings. Refusing to move at
        all answers a question nobody asked. So the dog walks to the nearest
        safe spot outside the zone -- the corridor by the stairwell -- and
        says plainly that this is as close as it goes.

        Returns (target, line) shaped like resolve()'s target, or None if the
        name is unknown.
        """
        entries = self.locations.get((name or "").strip().lower())
        if not entries:
            return None
        if floor is not None:
            entries = [e for e in entries if e["floor"] == floor]
            if not entries:
                return None
        best = min(entries, key=lambda e: (abs(e["floor"] - current_floor),
                                           math.dist(e["xy"], current_xy)))
        xy = self.nearest_free(best["floor"], best["xy"], clearance=APPROACH_CLEARANCE)
        hazard = self.hazard_at(best["xy"]) or ""
        return ({"floor": best["floor"], "xy": list(xy), "door_xy": list(xy)},
                f"I can take you to the corridor by the {name}, but no closer. {hazard}")

    def nearest_free(self, floor, xy, clearance=0.0):
        """The closest point to `xy` that a route may actually end on.

        Breadth-first, and it spreads over the *raw* grid rather than straight
        lines, so it cannot hop a wall: the first point 1 m clear of a
        stairwell as the crow flies is inside the room next door, which is no
        use to someone being led there. Spreading the way the dog would walk
        finds the corridor outside instead.

        A cell is accepted when it is free on the *planning* grid -- walls
        inflated by the robot's radius, no-go zones stamped in -- and at least
        `clearance` from any zone edge. Stopping the width of a paving slab
        from an open stairwell is not "near the stairs", it is on top of them,
        and the run-time stop would fire the moment the dog drifted.

        The one thing the spread may cross is the no-go zone it starts inside.
        A stairwell is drawn solid on the map -- "stairs" in locations.json is
        a point in the middle of a block of occupied cells with no free
        neighbour at all -- so a spread that only ever steps onto free cells
        never left the first cell, returned `xy` itself, and handed A* a goal
        inside a wall: every "take me to the stairs" died as `no route to
        stairs`, on all three floors, and with it the whole multi-stop demo.
        Inside that zone the spread ignores occupancy, because the zone is
        exactly the blob it is trying to get out of and its edge is where the
        corridor begins. Everywhere else walls still stop it, so the point it
        comes back with is one the dog could walk to.
        """
        raw, plan = self.grids[floor], self.plan_grids[floor]
        start = raw.world_to_grid(*xy)
        H, W = raw.grid.shape
        # The stop zones `xy` is in -- the ones whose insides are crossable
        # here. Not "any stop zone": crossing a stairwell two rooms away is
        # hopping a wall, which is the thing this spread exists to not do.
        blob = [z for z in self.behavior.zones
                if z.action_rules.speed_modifier <= 0.0 and z.contains_point(*xy)]

        def passable(r, c):
            """Free floor, or inside the zone the search started in."""
            if raw.is_free(r, c):             # walls stop the spread, zones do not
                return True
            x, y = raw.grid_to_world(r, c)
            return any(z.contains_point(x, y) for z in blob)

        seen, queue = {start}, deque([start])
        while queue:
            r, c = queue.popleft()
            here = plan.grid_to_world(r, c)
            if plan.is_free(r, c) and self.clear_of_zones(here, clearance):
                return here
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in seen or not (0 <= nxt[0] < H and 0 <= nxt[1] < W):
                    continue
                seen.add(nxt)
                if passable(*nxt):
                    queue.append(nxt)
        return tuple(xy)                      # nowhere free at all; caller will fail

    def clear_of_zones(self, xy, clearance):
        """Is this point at least `clearance` metres outside every stop zone?"""
        for zone in self.behavior.zones:
            if zone.action_rules.speed_modifier > 0.0:
                continue
            if zone.contains_point(*xy) or zone.distance_to_boundary(*xy) < clearance:
                return False
        return True

    def hazard_at(self, xy):
        """The announcement for a full-stop zone at this point, or None."""
        actions = self.behavior.query_actions(*xy)
        if actions["speed_modifier"] > 0.0:
            return None
        return (actions["announcements"] or ["That way is not safe"])[0]

    def resolve(self, name, current_floor, current_xy, floor=None, allow_hazard=False):
        """Pick the instance of `name` to go to.

        floor given (e.g. from cyberdog/language/floor_parser.py): only that
        floor's instance, or None if the place isn't on that floor.
        Otherwise: same floor first, else the closest floor.
        Returns None for unknown names.

        A place that sits inside a full-stop zone -- "take me to the stairs" --
        raises NoAccessibleRoute rather than resolving. It stays in
        locations.json because the zone is drawn around it and because a
        refusal should be able to name the thing it is refusing; what it is not
        is somewhere to lead a blind user. Internal lookups that know what they
        are doing pass allow_hazard=True.
        """
        entries = self.locations.get((name or "").strip().lower())
        if not entries:
            return None
        if floor is not None:
            entries = [e for e in entries if e["floor"] == floor]
            if not entries:
                return None
        best = min(entries, key=lambda e: (
            abs(e["floor"] - current_floor),
            math.dist(e["xy"], current_xy),
        ))
        hazard = None if allow_hazard else self.hazard_at(best["xy"])
        if hazard is not None:
            raise NoAccessibleRoute(
                f"I cannot take you to the {name}: that is a no-go area. {hazard}")
        return best

    def floors_of(self, name):
        """Floors a place exists on (empty list if unknown)."""
        return sorted({e["floor"] for e in self.locations.get((name or "").strip().lower(), [])})

    def transit(self, from_floor, to_floor, xy):
        """The lift serving both floors, or a refusal saying why there is none.

        Refusing is the right answer here, not falling back on the stairs. A
        floor with no lift is a floor this robot cannot take anyone to, and
        saying so is more useful than leading someone to a stair head.
        """
        served = set(self.floors_of(TRANSIT_NODE))
        missing = sorted({from_floor, to_floor} - served)
        if missing:
            raise NoAccessibleRoute(
                f"There is no lift serving floor {missing[0]}, and I do not "
                f"take stairs. I cannot get you to floor {to_floor} from here.")
        return self.resolve(TRANSIT_NODE, from_floor, xy, floor=from_floor)

    def lane(self, floor):
        """The path reshaping for one floor's legs: keep to lane_side."""
        if self.lane_side == "none":
            return None
        grid = self.plan_grids[floor]
        walls = self.lane_walls(floor)
        return lambda path: keep_side(path, grid, walls, self.lane_side,
                                      self.lane_distance, self.lane_width)

    def lane_walls(self, floor):
        """The floor's planning grid with doorways bridged -- lane.bridge_doors."""
        if floor not in self._lane_walls:
            self._lane_walls[floor] = bridge_doors(self.plan_grids[floor])
        return self._lane_walls[floor]

    def doorways(self, floor):
        """Boolean grid of the floor's doorway cells -- lane.doorway_cells."""
        return doorway_cells(self.plan_grids[floor], self.lane_walls(floor))

    def gates(self, floor, points):
        """The doorways a walk line goes through -- lane.door_gates."""
        return door_gates(points, self.plan_grids[floor], self.doorways(floor))

    def walk_line(self, floor, route):
        """(points, announcements per point): what the dog actually follows.

        The whole simplified polyline, not only its checkpoints. Checkpoints
        are where something is said, and they skip gentle bends -- one of
        which was the lane's move from the entrance door to the right-hand
        side, so the dog walked a 41 m diagonal across the corridor instead of
        the lane. Corners are rounded (lane.round_corners) so the person on
        the handle is not swung into what is beside the turn.
        """
        says = [[] for _ in route.polyline]
        for c in route.checkpoints:
            says[c.index] = list(c.announcements)
        points = [(float(x), float(y)) for x, y in route.polyline]
        if self.turn_radius <= 0:
            return points, says
        return round_corners(points, says, self.plan_grids[floor], self.turn_radius)

    def plan(self, start_floor, start_xy, target, lift_stop=None):
        """Returns a list of (floor, Route) legs, or None if any leg fails.

        `lift_stop` is where the dog waits outside the lift, if the caller
        knows: the leg to the lift ends there and the leg out starts there,
        instead of both at the car's centre. Cut back afterwards, a leg from
        the car lost its first lane point -- it is in the mouth of the shaft --
        and the dog walked a 41 m diagonal from the lift to the far end.
        """
        legs = []
        floor, xy = start_floor, tuple(start_xy)
        if target["floor"] != floor:
            lift = self.transit(floor, target["floor"], xy)
            stop = tuple(lift_stop) if lift_stop is not None else tuple(lift["xy"])
            leg = self.planner.plan_with_checkpoints(self.plan_grids[floor], self.behavior, xy,
                                                     stop, reshape=self.lane(floor))
            if leg is None:
                return None
            leg.checkpoints[-1].announcements = [f"Take the lift to floor {target['floor']}"]
            legs.append((floor, leg))
            floor, xy = target["floor"], stop

        leg = self.planner.plan_with_checkpoints(self.plan_grids[floor], self.behavior, xy,
                                                 tuple(target["xy"]), reshape=self.lane(floor))
        if leg is None:
            return None
        legs.append((floor, leg))
        return legs
