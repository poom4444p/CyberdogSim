# mapping — Ground Truth Map + Behavior Layer

**Layer 1** of the CyberDog Blind Navigation PoC.

## Purpose

The deterministic **"WHERE"** source: localization prior, global routes, semantic zone triggers.
No hardware required. No AI. Pure geometry + data structures.

## Components

| Module | Description |
|---|---|
| `occupancy_grid.py` | Load mesh/point cloud → 2D occupancy grid (ROS map_server compatible) |
| `behavior_layer.py` | Semantic zone annotations (grass, stairs, crosswalk) with point-in-polygon queries |
| `astar_planner.py` | A* pathfinding + Ramer–Douglas–Peucker simplification → checkpoints |

## Architecture Reference

From the CyberDog Skill spec:
- **Map decides WHERE** (3D map + Behavior Layer + A*)
- Output feeds → Task Planner (Layer 3)
- Acceptance: occupancy grid + A* route + checkpoints render correctly for ≥3 route queries

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a synthetic sample map
python scripts/mapping/generate_sample_map.py

# Visualize the map + zones + route
python scripts/mapping/visualize_map.py

# Run tests
python -m pytest tests/ -v
```

## Data Formats

- **Occupancy Grid**: `.pgm` + `.yaml` (ROS `map_server` format)
- **Behavior Zones**: JSON with polygon vertices + semantic tags
- **Route Output**: list of `(x, y)` waypoints + checkpoint metadata

## Where its files live

This layer's code is `src/cyberdog/mapping/`; the data and config it reads are
shared with the rest of the stack and resolved through `cyberdog.paths`.

```
src/cyberdog/mapping/
├── __init__.py          re-exports the three classes
├── occupancy_grid.py    OccupancyGrid
├── behavior_layer.py    BehaviorLayer, BehaviorZone
└── astar_planner.py     AStarPlanner

config/map_config.yaml   grid + A* settings (checkpoint spacing, turn threshold)
data/building/           the three-storey building: grids, meshes, locations, zones
data/samples/            sample_grid, sample_zones -- for the unit tests and demos
scripts/mapping/         rebuilding grids and meshes from point clouds
tests/                   test_occupancy_grid.py, test_behavior_layer.py, test_astar.py
```
