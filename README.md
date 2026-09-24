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

## Setup (macOS, Apple Silicon)

Two conda environments, on purpose — the same split the real robot has, where
the VLM is a service rather than an import. The twin runs without torch; only
the VLM server needs it. **The twin is the first environment and it is enough to
see the whole thing work.** The second is optional, and only for `--vamos`.

### 1. The twin — `cyberdog_sim`

```bash
conda create -n cyberdog_sim python=3.11 -y
conda activate cyberdog_sim

git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
pip install -e .
```

`mujoco_menagerie` supplies the Go2 robot model and deliberately lives outside
the repo — it is a large third-party asset, not this project's code.

**If you already have it somewhere else**, don't move it. Point at it instead,
and make the setting permanent for this environment only:

```bash
conda env config vars set MENAGERIE_PATH=~/Learn/mujoco_menagerie -n cyberdog_sim
conda activate cyberdog_sim     # re-activate for it to take effect
```

`paths.py` reads `MENAGERIE_PATH` and falls back to `~/mujoco_menagerie`.

### 2. Build the scene, then walk the dog

```bash
python -m cyberdog.sim.scene.build_scene --building
python -m cyberdog.sim.run_building "room 201"
```

That second command is the whole stack in one line: routing, the lift, voice
announcements, LiDAR, obstacle avoidance, and an MP4 written to
`output/scene_cache/building.mp4`.

> **Order matters.** `build_scene` writes the Go2's location into
> `building.xml` as an absolute path, so it must run *after* `MENAGERIE_PATH`
> is set. Move the menagerie later and you must rebuild the scene — changing
> the variable alone will not update the cached XML.

### 3. The VLM server — `vamos_mac` (optional)

Only needed for `--vamos`. Skip it on a first run; the dog navigates and avoids
obstacles without it (see [What is and isn't verified](#what-is-and-isnt-verified)).

```bash
conda env create -f vendor/VAMOS/environment_mac.yml
conda activate vamos_mac
pip install fastapi uvicorn pydantic python-multipart
```

That `pip install` line is not redundant: `vlm_server.py` imports FastAPI,
Uvicorn and Pydantic, and the upstream `environment_mac.yml` does not list them.

On macOS, conda's OpenMP runtime and the one bundled inside pip's torch collide
and abort the process on `import torch`. Set the documented escape hatch for
this environment:

```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n vamos_mac
conda activate vamos_mac
```

Start the server **from its own directory** — the script calls
`python vlm_server.py` by relative name and has no way to find itself:

```bash
cd vendor/VAMOS/server
bash start_server.sh
```

First run downloads the `mateoguaman/vamos` weights (several GB), so expect a
long pause before the port opens. Check it from another terminal:

```bash
curl -s http://127.0.0.1:8009/health && echo " — server up"
```

Then, back in `cyberdog_sim`:

```bash
python -m cyberdog.sim.run_building "room 201" --vamos
```

### 4. Check everything works, layer by layer

```bash
python tests/selftest.py          # every stage, ~2 min
python tests/selftest.py lidar    # or just one
pytest                            # the mapping layer's unit tests
```

---

## Running it

All commands run in `cyberdog_sim`, from the repo root.

| Command | What it does |
|---|---|
| `python -m cyberdog.sim.run_building "room 201"` | Full stack, three floors, with video |
| `python -m cyberdog.sim.run_building "room 201" --no-video` | Same, faster — no rendering |
| `python -m cyberdog.sim.run_building "room 201" --vamos` | VLM in the steering loop (needs the server) |
| `python -m cyberdog.sim.run_building "take me upstairs" --nlu` | Parse the command through the Gemma layer |
| `python -m cyberdog.sim.run_building "room 201" --speed 4` | Play the video back 4× faster (control still runs at 20 Hz) |
| `python -m cyberdog.sim.scene.build_scene --building` | Rebuild all three storeys |
| `python -m cyberdog.sim.scene.build_scene 2` | Rebuild one floor only |

Three of these are also on your `PATH` after `pip install -e .`:
`cyberdog-building`, `cyberdog-demo`, `cyberdog-record`.

---

## Troubleshooting

Four failures are common on a fresh Mac. Each one's **last** traceback line is
the diagnosis — the `File "..."` lines above it are only the call chain.

**`No module named 'cyberdog'`**
The package isn't installed in the active environment. Run `pip install -e .`,
and check you are in `cyberdog_sim` (`conda activate cyberdog_sim`). Note the
import path never starts with `src` — `pyproject.toml` declares
`where = ["src"]`, which makes `src/` the search root, so names begin at
`cyberdog`.

**`XML Error: Error opening file '.../mujoco_menagerie/unitree_go2/go2.xml'`**
The Go2 model isn't where the scene expects. Set `MENAGERIE_PATH` as in step 1,
then **rebuild the scene** — the stale path is baked into `building.xml`.

**`can't open file '.../vlm_server.py'`**
`start_server.sh` was run from the wrong directory. `cd vendor/VAMOS/server`
first, or use `(cd vendor/VAMOS/server && bash start_server.sh)` to leave your
shell where it is.

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`**
Two OpenMP runtimes in one process: conda's and the copy bundled in pip's torch.
Set `KMP_DUPLICATE_LIB_OK=TRUE` as in step 3. OpenMP calls this unsupported, and
it is — the realistic cost is thread contention, not wrong arithmetic, which is
an acceptable trade for a PoC inference server. The clean fix is installing
torch from conda-forge so one runtime serves everything.

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

A third environment exists for the command parser —
`pip install -e ".[language]"` adds torch, transformers and peft, and is what
`language/`, `main_planner.py` and the `--nlu` flag need. It can share
`cyberdog_sim` or stand alone.

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
  **This is why `--vamos` can perform worse than without it** on a floor the map
  already describes well: it replaces a globally-correct A* line with a 2 m
  horizon, and throttles speed by its own confidence. VAMOS earns its place
  where the map is wrong or missing, not where it is right.
- The collision counter is a **point test** on the dog's centre, not its body, so
  it scores a graze as clean. Margins in the passing runs were around 0.1 m of
  actual trunk clearance.
- Perception has **no memory** — each scan stands alone. Fine for a 360° sensor
  at 12 m, wrong the moment something is occluded.
- An obstacle parked against a wall is **invisible**: it falls inside the wall's
  own inflation radius and is discarded as already-explained.
