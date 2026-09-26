# Architecture

How the four layers actually connect, and why the boundaries fall where they do.
For what each layer contains, see its own README; for the specification this was
built against, see [`spec.md`](spec.md).

---

## The path a command takes

```
  "take me upstairs to the chemistry lab"
              |
              v
  +-------------------------------------------------------------+
  |  language/                                                   |
  |    command_splitter  "A, then B" -> ["A", "B"]   (rules)     |
  |    floor_parser      "upstairs"  -> floor + 1    (rules)     |
  |    infer             text -> {destination, floor}  (LoRA)    |
  +-------------------------------------------------------------+
              |  destination name + floor
              v
  +-------------------------------------------------------------+
  |  planning/building_router                                    |
  |    locations.json -> where is it, on which floor             |
  |    per-floor A* (mapping/) -> checkpoints                    |
  |    across floors -> legs joined by the lift                  |
  |    stairs -> NoAccessibleRoute (a refusal, spoken)           |
  +-------------------------------------------------------------+
              |  legs of checkpoints, in map coordinates
              v
  +-------------------------------------------------------------+
  |  sim/run_building   THE CONTROL LOOP, 20 Hz                  |
  |                                                              |
  |   every tick:                                                |
  |     lidar.scan()            -> ~2000 world points            |
  |     perception.update()     -> which of them the map cannot  |
  |                                explain                       |
  |     perception.free_destination() -> the goal to aim at      |
  |     control.advance()       -> which checkpoint we are on    |
  |                                                              |
  |   every 20 ticks (1 Hz), if --vamos:                         |
  |     checkpoint_projector.project_route()  -> goal PIXEL      |
  |     vamos_client.plan()     -> 5 candidate paths -> 1        |
  |       |- traversable()      cheap geometry test              |
  |       \- dreaming.factor()  imagined rollouts -> safety      |
  +-------------------------------------------------------------+
              |  target point
              v
        control law (control.py) -> velocity -> MuJoCo
```

---

## The two ideas that decide everything

### 1. Map decides WHERE, VLM decides HOW

`planning/` produces checkpoints in map coordinates.
`checkpoint_projector` turns the next one into a **pixel** in the dog's camera
frame. VAMOS is handed the image and that pixel, and answers with five candidate
paths *drawn on the image* — its job is the local manoeuvre, never the
destination.

This is why `free_destination` exists. VAMOS drives at whatever goal pixel it is
given. A* plans on a map with no crate in it, so the pixel can land *on* a
crate — and the model obligingly draws five paths into it (measured: at 0.8 m the
crate fills 60% of the frame and all five candidates still go straight through).
That is not a model that cannot avoid obstacles; it is a model being told to walk
into one. So the map's half of the division of labour has to use what the sensor
found, and hand VAMOS a goal on open floor.

### 2. AI proposes, simple code disposes

Nothing the VLM returns is followed on trust. Every candidate passes two
deterministic filters before it can steer:

| Filter | Question | Where |
|---|---|---|
| `traversable()` | Does the drawn line itself stay in free space? | `vamos_client.py` |
| `Dream.factor()` | If we actually chased it with our own controller, 24 times, with noise — how often do we survive, and with how much room? | `sensing/dreaming.py` |

The second returns `P(no collision) × (room left)`. Below `GATE = 0.5` the
candidate is rejected; above it, the factor also **throttles speed**, so a path
the dog only just believes it can walk is walked at half pace.

`dreaming.py` imports its gains from `sim/control.py` — the same constants the
live loop uses. A dream of a controller the dog does not have would tell us
nothing, so that shared import is load-bearing, not incidental.

---

## Clearance: the one number every safety decision reads

"Metres to the nearest thing the dog must not touch." Two sources, fused:

```
  clearance.py                 sensing/perception.py
  ------------                 --------------------
  occupancy grid walls         LiDAR returns the static
  + Behavior Layer             field cannot explain
    full-stop zones            + SHADOW behind each one
        |                              |
        +--------------+---------------+
                       v
              LiveClearance(x, y)  ->  min(static, detected)
```

`LiveClearance` has the same call signature as the static closure it replaced,
which is why the safety gate and the imagined rollouts needed no changes to start
seeing obstacles — they are handed this object instead and cannot tell the
difference.

Two details that are not obvious:

- **Zones matter as much as walls.** The grid is built from the floor mesh, so it
  is flat and knows nothing about the stairwell standing on it. Without the
  Behavior Layer's full-stop zones, a path into the stairs reads as wide open.
- **A range sensor sees surfaces, not solids.** A crate returns only the face
  pointing at the dog; the space behind that face is *unknown*, not free.
  Treating unknown as free is how a robot plans confidently into the inside of a
  box, so each return also shadows `SHADOW = 1.0 m` of the ray behind it. This is
  deliberately conservative and can over-claim when the real object is thin.

---

## Why the obstacles are not on the map

Every other solid in the scene — walls, slabs, lift, stairwells — is *also* in
the occupancy grid. That would make the entire perception half decorative: VAMOS
proposes paths from a picture, and those paths get scored against the same ground
truth the route already came from, so the camera could never contribute anything
the map did not already have.

`sim/scene/obstacles.py` is the exception, and the exception is the point. Its
crates are drawn into the scene and deliberately **absent from `data/building/`**
— no grid cell, no zone, no location. The map is stale on purpose. A* routes
straight through them; the static clearance field reports 0.40–0.55 m of open
floor exactly where they stand. That gap is what gives the perception layer a
job, and it is what `tests/selftest.py` measures.

The module asserts its own placement: each crate must reach the centreline the
dog actually walks (pure pursuit cuts corners, so long transits run down the
middle at y=9.5, not along the checkpoint polyline), and must leave at least
`MIN_GAP = 0.6 m` to squeeze past. A crate that sealed the corridor would be a
wall the planner cannot see, and every route on that floor would fail for reasons
no error message explains.

---

## Module dependencies

Arrows point from importer to imported. No cycles.

```
  main_planner ---> language ---> (torch)
       |
       +---------> planning ---> mapping
                      ^
                      |
  sim/run_building ---+
       |
       +--> sim/control ------> planning/building_router
       +--> sim/clearance ----> mapping/behavior_layer
       |                        sim/scene/build_scene
       +--> sim/overlay ------> planning/checkpoint_projector
       +--> sim/vamos_client -> (HTTP :8009)
       +--> sim/sensing/lidar ------> (mujoco)
       +--> sim/sensing/perception -> sim/clearance
       +--> sim/sensing/dreaming ---> sim/control
       +--> sim/robot/mujoco_robot -> sim/scene/build_scene
       +--> sim/scene/* -----------> cyberdog.paths

  everything --> cyberdog.paths     (filesystem locations, one place)
```

Two of these edges were recently *removed*, and the reasons generalise:

- `sensing/perception.py` used to do `from record_demo import clearance_field` —
  the live safety layer importing a video recorder. The clearance field now lives
  in `sim/clearance.py`.
- `sensing/dreaming.py` and `run_building.py` used to do
  `from run_demo import K_W, TURN_ONLY` — two modules reaching into a demo script
  for the control gains. Those now live in `sim/control.py`.

In both cases the shared thing was real and the location was wrong. If you find
yourself importing an entry point, that is the signal to extract.

---

## Timing budget

Measured by `tests/selftest.py latency` on an M-series Mac:

| Stage | Cost | Budget |
|---|---|---|
| LiDAR scan (1980 rays) | ~1.5 ms | — |
| Perception + `free_destination` | 2.4 ms | 50 ms (one 20 Hz tick) — 5% |
| `Dream.factor()` × 5 candidates | 7.4 ms | ~1.8 s (a VLM call) |
| Static clearance field (once per floor) | ~125 ms | cached — cannot be rebuilt per tick |

The control loop runs at 20 Hz; VAMOS is asked at 1 Hz, which is near its own
receding-horizon rate. The rate drops to 2 Hz (`REPLAN_BLOCKED`) while nothing
has cleared an obstacle, because the view changes as the dog closes in and a
candidate that was not there at 4 m often is at 2 m.
