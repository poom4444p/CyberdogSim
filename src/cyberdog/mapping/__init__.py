"""The ground-truth map: occupancy grids, semantic zones, and A* over them.

    OccupancyGrid   a floor's walls, from the FAST-LIO-style mesh
    BehaviorLayer   RDog's semantic zones -- stairs, grass, slow areas
    AStarPlanner    shortest path that respects both

Nothing here knows about the robot or the simulator; it is the map and the
search, and it is what the perception layer is measured against.
"""
from cyberdog.mapping.astar_planner import AStarPlanner
from cyberdog.mapping.behavior_layer import BehaviorLayer, BehaviorZone
from cyberdog.mapping.occupancy_grid import OccupancyGrid

__all__ = ["OccupancyGrid", "BehaviorLayer", "BehaviorZone", "AStarPlanner"]
