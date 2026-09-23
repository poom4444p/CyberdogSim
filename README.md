# Cyberdog Blind Navigation

A robotic guide dog: a navigation stack that takes a spoken command
("take me to room 201"), plans a route across a three-storey building, and walks
a quadruped there — announcing turns, refusing the stairs, and stepping around
obstacles that are not on any map.

Runs entirely in a MuJoCo digital twin. **No robot hardware required.**

Built on [VAMOS](https://arxiv.org/abs/2510.20818) (the vision-language path
planner), [RDog](https://dl.acm.org/doi/10.1145/3613904.3642227) (the semantic
map and handle interface), and Unitree Go2 kinematics. The full specification is
in [`docs/spec.md`](docs/spec.md); how the pieces fit together is in
[`docs/architecture.md`](docs/architecture.md).

**Design principle:** *AI proposes, simple code disposes.* Open-world
understanding lives in the AI modules. Every safety-critical decision is
deterministic.

---

## Quick start

```bash
# 1. The Go2 model (lives outside the repo)
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie

# 2. The stack
pip install -e .

# 3. Build the MuJoCo scene (three storeys, lift, stairwells, obstacles)
python -m cyberdog.sim.scene.build_scene --building

# 4. Walk the dog to a room on another floor, via the lift
python -m cyberdog.sim.run_building "room 201"
```

That last command is the whole stack in one line: routing, the lift, voice
announcements, LiDAR, obstacle avoidance, and an MP4 in `output/scene_cache/`.

Check everything works, layer by layer:

```bash
python tests/selftest.py          # every stage, ~2 min
python tests/selftest.py lidar    # or just one
pytest                            # the mapping layer's unit tests
```

---

## The four layers

A command travels through them in this order. Each has its own README with the
details.

| Layer | What it does | Read |
|---|---|---|
| [`language/`](src/cyberdog/language/README.md) | `"take me upstairs to the lab"` → structured destinations. gemma-2b-it + LoRA, with deterministic rules for splitting multi-stop commands and reading floor numbers. | `infer.py`, `command_splitter.py` |
| [`mapping/`](src/cyberdog/mapping/README.md) | The ground truth: occupancy grids per floor, RDog-style semantic zones (stairs, grass, slow areas), and A* that respects both. | `occupancy_grid.py`, `behavior_layer.py`, `astar_planner.py` |
| [`planning/`](src/cyberdog/planning/README.md) | Destination → route across floors, including which lift and which refusals. Then projects the next checkpoint into a camera pixel for the VLM. | `building_router.py`, `checkpoint_projector.py` |
| [`sim/`](src/cyberdog/sim/README.md) | The twin: the Go2, a virtual Mid-360 LiDAR, perception, imagined rollouts, and VAMOS in the steering loop. | `run_building.py`, `sensing/` |

`"map decides WHERE, VLM decides HOW"` is the division of labour between
`planning/` and the VLM, and it is worth knowing before reading either.

---

## Layout

```
src/cyberdog/
  paths.py            every filesystem location, in one place
  main_planner.py     command -> route, no simulator (needs [language])
  language/           the command parser
  mapping/            grids, zones, A*
  planning/           multi-floor routing + pixel projection
  sim/
    control.py        the control law and its gains
    clearance.py      static clearance from the map (walls + no-go zones)
    overlay.py        drawing onto the camera frame
    vamos_client.py   the VLM as a service: 5 candidates -> 1 chosen
    scene/            building the MuJoCo XML
    robot/            the Go2: kinematics, gait, pose
    sensing/          lidar -> perception -> dreaming
    run_building.py   ENTRY POINT: full stack, three floors
    run_demo.py       one floor, headless
    record_demo.py    one floor, with video

config/               map_config.yaml, camera_config.yaml
data/building/        grids, meshes, locations.json, zones.json
docs/                 spec.md, architecture.md
scripts/mapping/      rebuilding grids and meshes from point clouds
scripts/language/     interactive parser prompt
tests/                unit tests + selftest.py (the layer-by-layer harness)
vendor/VAMOS/         the upstream VAMOS repo and its VLM server

models/   datasets/   output/      generated or trained — all gitignored
```

Nothing the repo tracks is ever written into the source tree: built scenes and
recorded video go to `output/`, weights to `models/`, generated datasets to
`datasets/`.

---

## Environments

The stack is split across two Python environments on purpose — the same split
the real robot has, where the VLM is a service rather than an import.

| | Install | Runs |
|---|---|---|
| **Twin** | `pip install -e .` | Everything in `sim/`, `mapping/`, `planning/`. No torch. |
| **Language** | `pip install -e ".[language]"` | `language/`, `main_planner.py`. Needs torch, transformers, peft. |
| **VAMOS server** | `vendor/VAMOS/environment_mac.yml` | The VLM itself, as an HTTP service on `:8009`. |

To put VAMOS in the steering loop, start the server first, then pass `--vamos`:

```bash
bash vendor/VAMOS/server/start_server.sh
python -m cyberdog.sim.run_building "room 201" --vamos
```

Without `--vamos`, obstacle avoidance still runs — it comes from the LiDAR and
the clearance field, not from the VLM. See "What is and isn't verified" below.

---

## What is and isn't verified

Worth stating plainly, because the parts have different maturity.

**Verified against ground truth.** The crates in `sim/scene/obstacles.py` are
deliberately absent from the occupancy grid, so A* routes straight through them
and the perception layer has something real to find. `tests/selftest.py` scores
runs by comparing the dog's pose to those crates' footprints — data the robot is
never shown. Current results: 1741 unexplained LiDAR returns over the corridor
with **zero false positives**, and 4 of 5 routes arrive with no collision.

**Known limits, deliberately not yet closed:**

- The `room 101` route **stops short.** The floor-1 trolley blocks the same wall
  the route hugs, so getting past needs a full-corridor crossing — which needs a
  planned curve (CE-RRT\*, spec L6 s2). Stopping is the correct behaviour until
  that exists; driving through it would not be.
- **Zero-shot VAMOS cannot make the avoidance turn itself.** Its five candidates
  spread about ±0.25 m over a 2 m path; getting round a crate in this corridor
  takes about 0.65 m. So its paths are safe over their own length, pass the
  safety gate, and still lead into the obstacle. The map makes that turn instead
  (`perception.free_carrot`) until the LoRA fine-tune of spec L4 s5 exists.
- The collision counter is a **point test** on the dog's centre, not its body, so
  it scores a graze as clean. Margins in the passing runs were around 0.1 m of
  actual trunk clearance.
- Perception has **no memory** — each scan stands alone. Fine for a 360° sensor
  at 12 m, wrong the moment something is occluded.
- An obstacle parked against a wall is **invisible**: it falls inside the wall's
  own inflation radius and is discarded as already-explained.

---

## License

MIT
