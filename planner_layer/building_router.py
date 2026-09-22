"""Multi-floor routing on top of the 2D A* planner.

Each floor has its own occupancy grid (map_tools/data/building/maps/floorN/room_map.*).
Same floor: one A* call. Different floor: current floor -> stairs, then
stairs -> target on the target floor, with a "take the stairs" announcement
on the handoff checkpoint.

Running on Linux:
    Library module (imported by main_planner.py and test_locations.py), no GPU.
    Setup (once), from the project root:
        python3 -m venv .venv && source .venv/bin/activate
        pip install numpy pyyaml pillow scipy
    Maps must already exist in map_tools/data/building/maps/ (they're generated
    files; rebuild with map_tools/scripts/generate_building.py + mesh_to_grid.py,
    which additionally need `pip install open3d opencv-python`).
"""
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "map_tools", "src"))

from map_tools.occupancy_grid import OccupancyGrid  # noqa: E402
from map_tools.behavior_layer import BehaviorLayer  # noqa: E402
from map_tools.astar_planner import AStarPlanner  # noqa: E402

BUILDING_DIR = os.path.join(ROOT, "map_tools", "data", "building")
CONFIG_PATH = os.path.join(ROOT, "map_tools", "config", "map_config.yaml")


class BuildingRouter:
    def __init__(self, building_dir=BUILDING_DIR, config_path=CONFIG_PATH):
        with open(os.path.join(building_dir, "locations.json"), encoding="utf-8") as f:
            self.locations = json.load(f)
        self.floors = sorted({e["floor"] for entries in self.locations.values() for e in entries})
        self.grids = {
            n: OccupancyGrid.load(os.path.join(building_dir, "maps", f"floor{n}", "room_map"))
            for n in self.floors
        }
        self.behavior = BehaviorLayer.from_config(config_path)  # no zones inside the building
        self.planner = AStarPlanner.from_config(config_path)

    def resolve(self, name, current_floor, current_xy, floor=None):
        """Pick the instance of `name` to go to.

        floor given (e.g. from input_treating_layer/floor_parser.py): only that
        floor's instance, or None if the place isn't on that floor.
        Otherwise: same floor first, else the closest floor.
        Returns None for unknown names.
        """
        entries = self.locations.get((name or "").strip().lower())
        if not entries:
            return None
        if floor is not None:
            entries = [e for e in entries if e["floor"] == floor]
            if not entries:
                return None
        return min(entries, key=lambda e: (
            abs(e["floor"] - current_floor),
            math.dist(e["xy"], current_xy),
        ))

    def floors_of(self, name):
        """Floors a place exists on (empty list if unknown)."""
        return sorted({e["floor"] for e in self.locations.get((name or "").strip().lower(), [])})

    def plan(self, start_floor, start_xy, target):
        """Returns a list of (floor, Route) legs, or None if any leg fails."""
        legs = []
        floor, xy = start_floor, tuple(start_xy)
        if target["floor"] != floor:
            stairs = self.resolve("stairs", floor, xy)
            leg = self.planner.plan_with_checkpoints(self.grids[floor], self.behavior, xy, tuple(stairs["xy"]))
            if leg is None:
                return None
            leg.checkpoints[-1].announcements = [f"Take the stairs to floor {target['floor']}"]
            legs.append((floor, leg))
            floor, xy = target["floor"], tuple(stairs["xy"])

        leg = self.planner.plan_with_checkpoints(self.grids[floor], self.behavior, xy, tuple(target["xy"]))
        if leg is None:
            return None
        legs.append((floor, leg))
        return legs
