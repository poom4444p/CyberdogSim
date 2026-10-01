# Cyberdog Blind Navigation

A robotic guide dog: a navigation stack that takes a spoken command
("take me to room 201"), plans a route across a three-storey building, and walks
a quadruped there — announcing turns, refusing the stairs, stepping around
obstacles that are not on any map, and stopping to let people past.

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

## Start here: from a fresh clone to a walking dog

The repo holds code and the building map only. It does **not** include the
trained model, the training data, or any built scene, because those are
generated (they are gitignored):

| Folder | What goes there | Made by |
|---|---|---|
| `datasets/` | training data for the command parser | step 2 |
| `models/lora/` | the trained command parser | step 2 |
| `output/` | the built MuJoCo scene and recorded videos | step 3 |

So on a new machine, run the steps below in order. Every command runs from the
repo root. The steps are written for macOS on Apple Silicon; Linux notes are in
[`language/README.md`](src/cyberdog/language/README.md#linux--nvidia-gpu-users).

| Step | What | Needed? |
|---|---|---|
| [1](#1-install) | Install the environment and the robot model | yes |
| [2](#2-train-the-command-parser) | Train the command parser | only for plain-English commands (`--nlu`) |
| [3](#3-build-the-scene) | Build the 3D building | yes |
| [4](#4-walk-the-dog) | Walk the dog | yes |
| [5](#5-walk-the-dog-from-plain-english) | Walk the dog from plain English | needs step 2 |
| [6](#6-optional-the-vlm-server) | Start the VLM server | optional, only for `--vamos` |
| [7](#7-check-everything-works) | Run the tests | recommended |

### 1. Install

```bash
conda create -n cyberdog_sim python=3.11 -y
conda activate cyberdog_sim

pip install -e ".[language,dev]"   # the twin + the command parser + pytest

git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
```

- `pip install -e .` alone is enough for the twin. `[language]` adds torch,
  transformers, peft and trl, which step 2 needs; `[dev]` adds pytest.
- `mujoco_menagerie` supplies the Go2 robot model. It lives outside the repo
  because it is a large third-party asset.

**If you already have the menagerie somewhere else**, point at it instead of
moving it:

```bash
conda env config vars set MENAGERIE_PATH=~/path/to/mujoco_menagerie -n cyberdog_sim
conda activate cyberdog_sim     # REQUIRED: the variable only reaches a shell on activation
echo $MENAGERIE_PATH            # must print the path; if empty, activate again
```

### 2. Train the command parser

The command parser turns `"I need to pee"` into `restroom`. It is a LoRA
fine-tune of `unsloth/gemma-2b-it` on synthetic data the repo generates itself,
so there is nothing to download except the base model (not gated, fetched
automatically the first time).

```bash
python -m cyberdog.language.generate_dataset   # -> datasets/raw_dataset_english.json
python -m cyberdog.language.prepare_dataset    # -> datasets/gemma_training_data.jsonl
python -m cyberdog.language.train              # -> models/lora/
```

Training picks the fastest device by itself (CUDA, then Apple `mps`, then CPU)
and prints which one it chose. It keeps the checkpoint with the lowest
`eval_loss`, so the saved model is the best one, not just the last.

Then check the result:

```bash
python scripts/language/try_parser.py                  # type a command, see what it parses to
python -m cyberdog.language.evaluate --num-samples 100 # accuracy on fresh generated commands
python -m cyberdog.language.evaluate --num-samples 100 --held-out   # wording it never saw
```

`--held-out` is the number that matters: it only uses phrasings that were kept
out of training, so it shows whether the model generalises rather than
memorises.

Skip this step if you only want to name places exactly as the map does
(`restroom`, `cafeteria`, `room 201`). Everything else works without it. How
the parser works, what it recognises and how to add places or slang:
[`language/README.md`](src/cyberdog/language/README.md).

### 3. Build the scene

```bash
python -m cyberdog.sim.scene.build_scene --building
```

This writes the three-storey building to `output/scene_cache/building.xml`. It
bakes in the location of the Go2 model, so if you move `mujoco_menagerie`
later, run it again.

### 4. Walk the dog

```bash
python -m cyberdog.sim.run_building "room 201"
```

That is the whole stack in one line: routing, the lift, voice announcements,
LiDAR, obstacle avoidance, and a video written to
`output/scene_cache/building.mp4`. Add `--no-video` for a faster run.

At the end it prints a score:

```
ARRIVED on floor 1 at (33.1, 4.5) after 32s of sim time
obstacles: up to 20 returns the map could not explain, 0.0s crawling, ...
collisions: none -- the dog never entered an obstacle's footprint
people: 1 times it stopped to let someone past, 1.1s waiting in total
contact: none -- closest it came to anybody was 3.31 m
```

Read `collisions` and `contact` first: they are scored against ground truth the
robot is never shown. A run that arrives but reports `CONTACT` is not a pass.

### 5. Walk the dog from plain English

With the model from step 2, add `--nlu`:

```bash
python -m cyberdog.sim.run_building "I need to pee" --nlu
python -m cyberdog.sim.run_building "I need to pee on the 2nd floor" --nlu
python -m cyberdog.sim.run_building "I need to pee, then take me to the cafeteria" --nlu
python -m cyberdog.sim.run_building "take me upstairs" --nlu
```

And with people walking the corridors, which are on no map:

```bash
python -m cyberdog.sim.run_building "I need to pee" --nlu --pedestrians 3 --seed 1
```

### 6. (Optional) The VLM server

Only needed for `--vamos`, which puts the VAMOS vision-language model in the
steering loop. The dog navigates and avoids obstacles without it. It runs in a
second environment, the same split the real robot has, where the VLM is a
service rather than an import.

```bash
conda env create -f vendor/VAMOS/environment_mac.yml
conda activate vamos_mac
pip install fastapi uvicorn pydantic python-multipart   # not in the upstream yml
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n vamos_mac
conda activate vamos_mac                                 # again, to pick that up

cd vendor/VAMOS/server       # must start from here
bash start_server.sh
```

The first start downloads the `mateoguaman/vamos` weights (several GB), so
expect a long pause. From another terminal, check it is up, then run with it:

```bash
curl -s http://127.0.0.1:8009/health && echo " — server up"

conda activate cyberdog_sim
python -m cyberdog.sim.run_building "room 201" --vamos
```

`--vamos` can score *worse* than without it on a floor the map already solves;
see [What is and isn't verified](#what-is-and-isnt-verified) before reading a
bad run as a bug.

### 7. Check everything works

Run these from the repo root, in `cyberdog_sim` -- **not** `vamos_mac`. If you
just started the VLM server (step 6), that terminal is still in `vamos_mac`;
open a new one or switch back first.

```bash
conda activate cyberdog_sim
python tests/selftest.py          # every stage of the stack, ~2 min
python tests/selftest.py lidar    # or just one stage
pytest                            # unit tests
```

Or, without changing the active environment:

```bash
conda run -n cyberdog_sim --no-capture-output python tests/selftest.py
```

Stages run bottom-up and can be named one at a time: `scene`, `lidar`,
`perception`, `destination`, `crowd`, `latency`, `dreaming`, `vamos`, `run`.
Each check prints `[PASS]` or `[FAIL]` with the number it judged on, and the
run ends with `ALL PASS` or the list of failures. The first failure is usually
the real one -- a bad scene fails every stage above it.

- The `vamos` stage uses a stand-in server, so it needs neither the VLM server
  nor the `vamos_mac` environment.
- `pytest` skips `selftest.py` on purpose (see `pyproject.toml`); run it
  directly as above.

---

## Troubleshooting

Each traceback's **last** line is the diagnosis — the `File "..."` lines above
it are only the call chain.

**`No module named 'cyberdog'`**
The package isn't installed in the active environment. `conda activate
cyberdog_sim`, then `pip install -e .`. Import names start at `cyberdog`, never
`src`. Most often the wrong environment is active -- `vamos_mac` after starting
the VLM server. `python -c "import sys; print(sys.executable)"` should print a
path inside `.../envs/cyberdog_sim/` (`which python` can show a pyenv shim even
when the right interpreter runs).

**`No module named 'torch'` (with `--nlu` or during training)**
The command parser's extra is missing: `pip install -e ".[language]"`.

**`--nlu` fails with an error naming `models/lora` or `adapter_config.json`**
The trained model is not in the repo. Train it (step 2).

**`XML Error: Error opening file '.../mujoco_menagerie/unitree_go2/go2.xml'`**
The built scene points at a Go2 model that is not there. Either the menagerie
moved, or `MENAGERIE_PATH` is set in the conda config but not in *this shell*.
`echo $MENAGERIE_PATH`; if it prints nothing, `conda activate cyberdog_sim`,
then rebuild the scene (step 3). The path in the error is the one that was
*looked for* (the default `~/mujoco_menagerie` when the variable is missing),
not where your copy is. `build_scene` checks this now and refuses with an
explanation instead of writing a broken scene.

**`no destination found in '...'`**
You named a place the map does not carry — `"pee"` rather than `"restroom"`.
Without `--nlu` the destination must match a name in
`data/building/locations.json`. Name the place, or add `--nlu`.

**`can't open file '.../vlm_server.py'`**
`start_server.sh` was run from the wrong directory. `cd vendor/VAMOS/server`
first.

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`**
Two OpenMP runtimes in one process (conda's and the one inside pip's torch).
Set `KMP_DUPLICATE_LIB_OK=TRUE`, as in step 6. The repo's own scripts set it
for you. The cost is thread contention, not wrong results; the clean fix is
torch from conda-forge so one runtime serves everything.

**`OMP: Warning #20: KMP_DUPLICATE_LIB_OK="": Wrong value, boolean expected.`**
The variable is set but empty in this shell. `export KMP_DUPLICATE_LIB_OK=TRUE`,
or re-activate the conda environment.

---

## Command reference

All commands run in `cyberdog_sim`, from the repo root. Flags combine freely.

| Command | What it does |
|---|---|
| `python -m cyberdog.sim.run_building "room 201"` | Full stack, three floors, with video |
| `python -m cyberdog.sim.run_building "room 201" --no-video` | Same, faster — no rendering |
| `python -m cyberdog.sim.run_building "room 201" --vamos` | VLM in the steering loop (needs the server) |
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3` | People walking the corridors, on no map — the dog stops for them |
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --seed 4` | Same, a different crowd (a seed is reproducible) |
| `python -m cyberdog.sim.run_building restroom --handle "tug@10,continue@15"` | The person on the handle: a tug at 10 s stops the dog until continue at 15 s; `pull@T1-T2` slows it, `push@T1-T2` undoes a pull. With `--handle`, a continue is also what answers the lift prompt, in place of `--auto-confirm` (`sim/handle.py`) |
| `python -m cyberdog.sim.run_building "take me upstairs" --nlu` | Parse the command through the Gemma layer (needs `.[language]`) |
| `python -m cyberdog.sim.run_building "room 201" --speed 4` | Play the video back 4× faster (control still runs at 20 Hz) |
| `python -m cyberdog.sim.scene.build_scene --building` | Rebuild all three storeys |
| `python -m cyberdog.sim.scene.build_scene 2` | Rebuild one floor only |

Three of these are also on your `PATH` after `pip install -e .`:
`cyberdog-building`, `cyberdog-demo`, `cyberdog-record`.

Without `--nlu`, the destination must be a name from
`data/building/locations.json`. Splitting multi-stop commands and reading floor
phrases ("on the 2nd floor") are rules, so they work with or without `--nlu`;
only slang like `"I'm starving"` needs the model.

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

## What is and isn't verified

Worth stating plainly, because the parts have different maturity.

**Verified against ground truth.** The crates in `sim/scene/obstacles.py` are
deliberately absent from the occupancy grid, so A* routes straight through them
and the perception layer has something real to find. `tests/selftest.py` scores
runs by comparing the dog's pose to those crates' footprints — data the robot is
never shown. Current results: 2008 unexplained LiDAR returns over the corridor
with **zero false positives**, and all eight scored routes end as intended.
Across every named destination in the building — all three floors — **every
route arrives with zero collisions**, including `chemistry lab`, which used to
stop short of its own door. The stairs, correctly, are not one of them: those
end in the corridor outside, with the refusal spoken. Nothing in the building
drives through anything any more.

The suite ran six routes until recently and reported `ALL PASS` for three
successive layouts of this scene, one of which drove through a crate. None of
its six came at a box from the side the box's clearance was *not* on. It now
runs eight, including both sides of the floor-3 cartons, and scores each route
against what it is known to do rather than a single arrived/not flag.

The same is true of the people in `sim/scene/pedestrians.py`, who are on no map
either and who additionally move, so a single scan cannot describe them.
`sensing/tracking.py` gives a thing a velocity, and the dog stops for anything
walking whose path meets its own inside the next 2.5 s. Scored the same way:
six seeds of three people on the floor-2 route, **6 of 6 arrive with no
collision and no contact** — the closest anybody came was 0.55 m, and four of
the six stayed beyond 0.6 m. That is the first time this project has measured
zero: the documented figure was 2 of 6 seeds inside the 0.55 m threshold, and
the obstacle resize briefly made it 3 of 6. What cleared it was not the scene
but `free_destination` finishing its sideways crossing *at the obstacle*
instead of out at the destination, so the dog stops arriving alongside a crate
still half-way across and squeezing past in the lane people walk down.
With the yielding disabled and everything else identical, the same
seeds produce contact throughout — which is what says the stopping is doing the
work rather than the crowd happening to miss. In an empty building the same logic
yields **zero** times, which took two shape filters to reach: a crate's visible
face slides along itself at 0.6 m/s as the dog walks past, which is a walking
pace, so speed alone cannot tell them apart.

**Known limits, deliberately not yet closed:**

Three things that used to be on this list are not any more, and all three were
the same class of bug — a test that asked a question other than the one that
mattered. They are written up in `sim/sensing/perception.py` and
`sim/run_building.py`; briefly:

- **Obstacle avoidance ran only while the camera could see the goal pixel**
  (`if state["state"] == "TRACK"`). `ALIGN` is what the projector reports when
  the dog is turning into a doorway, so for the whole of every turn the dog
  steered at the raw A\* waypoint with the LiDAR ignored. That was the
  `chemistry lab` collision: 0.4 s inside the cartons, every tick of it in
  `ALIGN`. The Mid-360 is a 360° sensor; whether a projection is in frame says
  nothing about what is in front of the dog.
- **It drew straight lines through walls.** Within 2 m of a corner checkpoint,
  `pick_destination` interpolates *past* the corner, so the goal handed to
  `free_destination` sat metres inside a room. The tightest point on the line to
  it is then the wall, every sideways offset lands inside the building, and the
  verdict is "I cannot find a way past this" — measured with the dog standing in
  a corridor with 0.68 m clear on every side.
- **The sideways crossing finished in the wrong place.** The offset was measured
  at the obstacle and applied at the destination, so the dog drove a shallow
  diagonal and arrived *level with* the crate only part-way across — told to
  cross 0.70 m, it reached the cartons with 0.24 m against the 0.20 m it
  insists on. This is what produced the pedestrian grazes: squeezing past a
  crate at the last moment, in the lane people walk down.

Together they removed every collision in the building, took pedestrian contacts
from 2-of-6 seeds to **0 of 6**, and fixed the `restroom --seed 5` timeout that
this list used to carry. No threshold changed: `DESTINATION_CLEAR`, `LINE_NEED`
and `STATIC_MIN` are what they were, having been measured first and found not
to be binding.

- **A detour is committed to one side now, and that was the real limit.**
  `chemistry lab` stopped short of its own door for a long time, and the
  diagnosis was geometric: cartons 1.2 m east of the door, so getting in means
  passing them on the north and turning 90° south immediately — a curve, which
  a goal displaced sideways cannot express. Half of that was true and half of
  it was a flip-flop: "which side has more room" is judged afresh every tick
  from a scan that changes as the dog closes in, and level with an obstacle the
  answer alternates. The dog leant north, was sent south the next tick, and
  undid its own crossing. With the side committed until the obstacle is out of
  sight (`free_destination(prefer=...)`), `chemistry lab` and the floor-2
  `restroom` both arrive. CE-RRT\* (spec L6 s2) is still what a real curve
  wants; it is no longer what these two routes were waiting for.
- **Zero-shot VAMOS proposes short paths, and whether that is enough depends on
  the corridor.** Its five candidates spread about ±0.25 m over a 2 m path. When
  an obstacle demanded 0.65 m of sidestep — the old 1.45 m-deep boxes — its paths
  were safe over their own length, passed the safety gate, and still led into the
  crate, and `room 201` did not finish under `--vamos` in four consecutive runs.
  With realistically sized boxes the sidestep is inside what the model proposes:
  the same route now arrives in 92 s and the gate rejects 15 of 95 candidates
  instead of 113 of 149, at a mean safety of 1.00. Nothing about the VLM changed — same
  server, same prompt, same gate — so read that as a fact about this corridor,
  not about zero-shot VAMOS. Where a turn does exceed its horizon the map still
  makes it (`perception.free_destination`), until the LoRA fine-tune of spec
  L4 s5 exists. **`--vamos` can still perform worse than without it** on a floor
  the map already describes well: it replaces a globally-correct A* line with a
  2 m horizon and throttles speed by its own confidence, costing 11 s on that
  route.
- **Floor 1 is a lobby, not a corridor, on any cross-floor route.** `main
  entrance` is 1.6 m from the lift, so the floor-1 leg of the `building.mp4`
  demo is five metres of turning out of a doorway: VAMOS is called there but
  nearly every candidate goes through the entrance wall, so its paths are barely
  on screen. Give it a floor-1 destination (`room 101`, `cafeteria`) to watch it
  steer down a corridor.
- `--vamos` **is reproducible with `--vlm-latency` fixed.** The client asks for
  its 5 candidates by beam search (`num_beams=5`, `temperature=0`) rather than
  the server's default sampling at `temperature=1.0`, which gave 5 different
  paths for the same frame on every call. Beam search gives the same 5, still
  distinct, at the same 1.6 s. `second floor restroom --nlu --vamos
  --pedestrians 4 --seed 2 --vlm-latency 1.8` now repeats tick for tick (same
  `commands:` sha256). Without `--vlm-latency` an answer lands after the
  call's measured time, which varies, so runs still drift. `--vamos-sample`
  goes back to sampling.
- **VAMOS answers arrive late, and the loop does not wait for them.** A call
  takes about 1.8 s against the real server. It used to block the control loop
  while the twin paused the world, which hid that a real dog would walk on
  blind for that long. Now a request is sent from one tick's image and its
  answer lands `--vlm-latency` seconds of sim time later (default: the call's
  measured time). Meanwhile the 50 ms loop keeps its LiDAR, stop and yield. A late
  answer is cut to what is still ahead of the dog and re-dreamed from where
  it is now, so a path that has run into something since is rejected on
  arrival. Runs print `VLM ASYNC:` with how old the answers were and how far
  the dog had moved. The `--vamos` results above predate this.
- **Collisions are whole-body, and routes keep right.** `COLLISIONS:` is
  MuJoCo's contacts between the whole Go2 body and the walls, crates, lift and
  stairs. It used to be a point test on the dog's centre against the crates
  only, which hid the dog scraping the south corridor wall on `restroom`,
  `room 101` and `room 201` for up to 13 s. A\*'s shortest line ran along the
  edge of the free space, 0.25 m from the wall face. Three fixes, and all 8
  routes are now clean:
  - Routes keep to the right-hand side of corridors, 0.75 m from the wall, as
    pedestrians do in Denmark (`lane` in `config/map_config.yaml`,
    `mapping/lane.py`). Rooms and lobbies have no lane.
  - Line simplification and checkpoint extraction never replace a stretch of
    route with a straight line that leaves free space. The dropped doorway
    point was cutting `room 101`'s door frame.
  - Avoidance follows the route round a corner when the route itself is
    clear, instead of detouring from a straight line that cuts across an
    obstacle. That line walked `chemistry lab` into the cartons once it kept
    right.
- **The person on the handle is scored.** They are modelled 1.1 m straight
  behind the dog on a rigid handle, with a 0.25 m radius. `HANDLER:` reports
  their contact with walls, crates, lift and stairs. On a turn of radius r
  they swing √(r² + 1.1²) − r outside the dog's path, so a sharp corner puts
  them into whatever is beside it. Contact started on 7 of 8 routes and is now
  **0 of 8**, and the self-test fails on any (`person on the handle
  untouched`).

  What fixed it: the dog follows the whole route, not only its announcement
  checkpoints (these skipped the lane's bends). Corners are arcs of up to
  1.5 m radius (`lane.turn_radius`). The lane is 0.75 m from the wall, to
  leave room for the swing. Legs to and from the lift start at the waiting
  point outside it. Across a lobby the route blends into the lane instead of
  making an S-bend. Then three things in `run_building.py`:
  - **Turns are checked for the person** (`spare_handler`). Every contact
    left was a turn: a pivot beside the crate just gone round, a left turn
    past the lift. Each command is rolled forward 0.8 s; where it would bring
    the person within 0.35 m of a wall or a detected obstacle, the turn is
    eased and the dog keeps walking, so the arc widens. Walls are measured to
    their real faces, not the planner's inflated ones.
  - **The lift shaft's side walls are on the map** (`lift.SHAFT_WALLS`, folded
    into `clearance.py`). They were in the scene and not the grid, 0.5 m into
    floor the map called open, and the LiDAR's returns off them were thrown
    away as "already mapped".
  - **Doors are entered on their axis** (`funnel`). From 2 m out the dog
    steers onto the doorway's centre line, and in the last metre it lines up
    before it walks. A dog off the route -- round a person, or on a VAMOS
    path -- used to reach the frame 0.5 m off-centre and pivot there, which
    swung the person into the floor-2 crate beside the restroom door.
    Reshaping the route to meet doors square was tried and made it worse for
    doors on the lane's own side of the corridor; the funnel alone does it.
- **People.** Pedestrians keep right, and cross the corridor all run long,
  from the left and the right, straight or on a slant. Before, every walker
  became a corridor-walker after its first leg, so nobody near the dog was
  crossing at all, and up to half came head-on down the dog's own side.
  Someone the dog is standing in front of steps round it after 2 s
  (`GIVE_WAY`), which ends standoffs where each waited for the other. When
  the dog goes round someone who has stopped, it keeps 0.6 m from them, not
  a crate's 0.35 m, and walks at 0.4 m/s.
  It does **not** step back first: that was tried, and reversing pushes the
  rigid handle into the person holding it.

  Over 40 seeds of `room 201 --pedestrians 3`, all arrive and the dog touches
  nothing. The person on the handle touches something on 6 (5.8 s in all,
  every one a wall -- never a box), and 7 seeds come within 0.55 m of a
  pedestrian (8.5 s in all). Before the fixes below: 7 seeds and
  7.5 s on the handle, 5 of them against the cart or crate, and 13 seeds and
  18.3 s of pedestrian contact.
  - **Easing a turn keeps clear of people** (`SWING_PEOPLE`). It walks on
    where the dog would have pivoted, and on seeds 4 and 12 that walked it
    towards somebody standing beside it. A step easing adds keeps 0.7 m from
    where each person-sized track is heading, or the turn goes ahead.
  - **No avoidance goal behind a wall.** A point shifted 1.6 m sideways out
    of the corridor is open floor in the room behind it, and the line to it
    was only checked against what the LiDAR found; seed 8 walked into the
    floor-2 north wall for 6.5 s. Lines may no longer enter a wall, tested
    against the real wall faces (`clearance.wall_face_field`).
  - **Straight past a box leaves room for the person** (`PERSON_LINE`). The
    dog passed the floor-2 cart and crate 0.23 m clear, enough for itself;
    the person follows the same line 1.1 m behind and is 0.25 m across the
    shoulders. A line now counts as clear, no detour needed, only with 0.35 m
    to spare; with less, the dog leans away. Rounding a corner still needs
    only the dog's 0.20 m (`LINE_NEED`).
  - **The front of the dog is guarded** (`NOSE_MIN`, `SWING_NOSE`). The
    proximity stop reads the centre, and the Go2 is twice as long as it is
    wide: seed 8 kept 0.25 m at the centre and put a front leg on the cart's
    corner. Walking on is refused while either front corner is within 0.15 m
    of something and closing -- the dog turns first -- both in the command
    and in an eased one.

  What is left is not boxes: people walking into a dog that is standing and
  waiting, going round somebody who has stopped, overtaking in the same
  lane -- people heading west keep to the north lane (y = 10.40), which is
  also the dog's lane (y ≈ 10.10), and pedestrians do not yet overtake on
  the left -- and the person against a wall on a turn round somebody.
- Perception has **no memory** — each scan stands alone. Fine for a 360° sensor
  at 12 m, wrong the moment something is occluded. Tracking adds half a second
  of it, enough for a velocity and not enough to survive an occlusion: someone
  who steps behind a crate is a lost track.
- The dog **stops** for a person rather than flowing around one. Spec L6 wants a
  pedestrian avoided without a full stop, which needs a planner that can commit
  to a curve; until CE-RRT\* exists, stopping is the honest version, and every
  run counts its stops.
- The pedestrians are capsules on scripted legs, not a pedestrian model. They
  know one thing about the dog — do not walk into what is in front of you — and
  making them cleverer would quietly solve the robot's problem for it.
- An obstacle parked against a wall is **invisible**: it falls inside the wall's
  own inflation radius and is discarded as already-explained.
