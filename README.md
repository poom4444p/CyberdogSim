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

## Setup (macOS, Apple Silicon)

Two conda environments, on purpose — the same split the real robot has, where
the VLM is a service rather than an import. The twin runs without torch; only
the VLM server needs it. **The twin is the first environment and it is enough to
see the whole thing work.** The second is optional, and only for `--vamos`.

`--nlu` is the exception to the split: the Gemma command parser is small enough
to run in-process, so it wants torch inside the twin rather than a server of its
own. It is an optional extra on the same environment — step 1 has it.

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
conda activate cyberdog_sim     # REQUIRED: see below
```

`paths.py` reads `MENAGERIE_PATH` and falls back to `~/mujoco_menagerie`.

> **That second line is not decoration.** `conda env config vars set` writes the
> variable into the *environment's config*; it reaches a shell only when that
> environment is activated. Set it and carry on in the shell you are already
> in, and the variable is configured and still absent — `echo $MENAGERIE_PATH`
> prints nothing, and every command you run uses the fallback. Re-activate even
> if the prompt already says `(cyberdog_sim)`.

`echo $MENAGERIE_PATH` is the check. If it prints nothing, the next section is
about to bite.

**For `--nlu` only**, add the command parser's dependencies. Without them the
twin still runs; you just have to name a place the map knows, because matching
is done against `locations.json` by name rather than by model:

```bash
pip install -e ".[language]"
```

That pulls torch, transformers and peft into `cyberdog_sim`. The LoRA adapter is
already in `models/lora`, and the base model (`unsloth/gemma-2b-it`, not gated)
is fetched once on first use. No `KMP_DUPLICATE_LIB_OK` is needed here —
`language/infer.py` sets it for you.

### 2. Build the scene, then walk the dog

```bash
python -m cyberdog.sim.scene.build_scene --building
python -m cyberdog.sim.run_building "room 201"
```

That second command is the whole stack in one line: routing, the lift, voice
announcements, LiDAR, obstacle avoidance, and an MP4 written to
`output/scene_cache/building.mp4`.

> **The scene caches the path, so order matters.** `build_scene` writes the
> Go2's location into `building.xml` as an absolute path. It must therefore run
> *after* `MENAGERIE_PATH` is in the shell, and if you move the menagerie later
> you must rebuild the scene — changing the variable alone will not touch the
> cached XML.
>
> This used to fail in the most confusing way available. The builder wrote
> whatever path it had and reported success; the complaint arrived later, from
> MuJoCo, out of whichever run first loaded the file, naming a path you never
> typed on a line you never wrote. Rebuilding the scene is the obvious
> response and silently re-bakes the same wrong path, so the identical error
> comes back and the fix appears not to work.
>
> `build_scene` now checks before it writes, so the failure lands where the
> mistake is and says what to do about it:
>
> ```
> cannot find the Go2 model at /Users/you/mujoco_menagerie/unitree_go2/go2.xml
>   looked there because of the default, because MENAGERIE_PATH is not set in this shell
>   fix: conda env config vars set MENAGERIE_PATH=/path/to/mujoco_menagerie -n cyberdog_sim
>        then `conda activate cyberdog_sim` again -- the variable only reaches a shell on activation
> ```

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
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3` | People walking the corridors, on no map — the dog stops for them |
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --seed 4` | Same, a different crowd (a seed is reproducible) |
| `python -m cyberdog.sim.run_building "take me upstairs" --nlu` | Parse the command through the Gemma layer (needs `.[language]`) |
| `python -m cyberdog.sim.run_building "room 201" --speed 4` | Play the video back 4× faster (control still runs at 20 Hz) |
| `python -m cyberdog.sim.scene.build_scene --building` | Rebuild all three storeys |
| `python -m cyberdog.sim.scene.build_scene 2` | Rebuild one floor only |

Three of these are also on your `PATH` after `pip install -e .`:
`cyberdog-building`, `cyberdog-demo`, `cyberdog-record`.

### Saying it in plain English — `--nlu`

Without `--nlu` the destination is matched against `locations.json` by name, so
it has to *be* one of the names: `restroom`, `cafeteria`, `room 201`. Ask for
`"pee"` and you get `no destination found` — the rule matcher has never heard of
it, and is not supposed to have. Slang is the fine-tune's job, and `--nlu` is
what puts it in the loop.

| Command | What the parser does with it |
|---|---|
| `python -m cyberdog.sim.run_building "I need to pee" --nlu` | slang → `restroom` |
| `python -m cyberdog.sim.run_building "I need to pee on the 2nd floor" --nlu` | slang → `restroom`, floor rules → floor 2 |
| `python -m cyberdog.sim.run_building "I need to pee, then take me to the cafeteria" --nlu` | splitter → two stops, `restroom` then `cafeteria` |
| `python -m cyberdog.sim.run_building "I'm starving" --nlu` | slang → `cafeteria` |

Only the destination needs the model. Splitting multi-stop commands and reading
floor phrases are rules either way — that is where they live in the pipeline —
so `"on the 2nd floor"` is handled the same with `--nlu` or without it.

### All three at once

The flags compose, and this is the one worth watching: a spoken command, a
corridor with people in it, and the VLM steering.

```bash
# slang + a crowd — no server needed
python -m cyberdog.sim.run_building "I need to pee" --nlu --pedestrians 3 --seed 1

# ...and with VAMOS in the steering loop (start the server first, step 3)
python -m cyberdog.sim.run_building "I need to pee" --nlu --pedestrians 3 --seed 1 --vamos

# the same, rendered 4× faster, for a quick look
python -m cyberdog.sim.run_building "I need to pee" --nlu --pedestrians 3 --seed 1 --speed 4
```

Each prints its own scoring at the end — whether it arrived, how near it came to
a crate, how often it stopped for somebody, and how close it got to them:

```
ARRIVED on floor 1 at (33.1, 4.5) after 32s of sim time
obstacles: up to 20 returns the map could not explain, 0.0s crawling, ...
collisions: none -- the dog never entered an obstacle's footprint
people: 1 times it stopped to let someone past, 1.1s waiting in total
contact: none -- closest it came to anybody was 3.31 m
```

Read `collisions` and `contact` first: they are scored against ground truth the
robot is never shown. A run that arrives but reports `CONTACT` is not a pass.

**`--vamos` can score worse than without it** on a floor the map already solves
— see [What is and isn't verified](#what-is-and-isnt-verified) before reading a
bad run as a bug.

---

## Troubleshooting

Six failures are common on a fresh Mac. Each one's **last** traceback line is
the diagnosis — the `File "..."` lines above it are only the call chain.

**`No module named 'cyberdog'`**
The package isn't installed in the active environment. Run `pip install -e .`,
and check you are in `cyberdog_sim` (`conda activate cyberdog_sim`). Note the
import path never starts with `src` — `pyproject.toml` declares
`where = ["src"]`, which makes `src/` the search root, so names begin at
`cyberdog`.

**`XML Error: Error opening file '.../mujoco_menagerie/unitree_go2/go2.xml'`**
A stale absolute path baked into `building.xml`, which is a cached build
artifact, not code. Two causes, and the second is the one that wastes an
afternoon:

- The menagerie moved. Rebuild the scene.
- `MENAGERIE_PATH` is set in the environment's config but not in *this shell*,
  because the shell has not been activated since. Then rebuilding does not help
  — it writes the same wrong path again, and the same error returns unchanged.
  `echo $MENAGERIE_PATH`; if it is empty, `conda activate cyberdog_sim` and
  rebuild. Building now refuses outright in this case rather than writing a
  broken scene, so a fresh checkout gets the explanation instead of the
  traceback.

Note the path in the message is the one that was *looked for*, which is the
fallback `~/mujoco_menagerie` when the variable is missing — it is not where
your copy is, and chasing it is the wrong trail.

**`no destination found in '...' (try --nlu, or name a place from locations.json)`**
You asked for somewhere by a name the map does not carry — `"pee"` rather than
`"restroom"`. Without `--nlu` the destination is a literal match against
`locations.json`, which has no slang in it by design. Either name the place, or
add `--nlu` to put the fine-tune in the loop. If `--nlu` then fails on
`No module named 'torch'`, the command parser's extra is not installed: see
step 1, `pip install -e ".[language]"`.

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
- `--vamos` is **not reproducible.** The VLM server samples at `temperature=1.0`
  and the client sends neither a temperature nor a seed, so two identical
  commands give different paths. `--seed` pins the crowd, not the model. The two
  `room 201` runs above differed by 2 calls and 16 candidates on the same route.
- The collision counter is a **point test** on the dog's centre, not its body, so
  it scores a graze as clean. Margins in the passing runs were around 0.1 m of
  actual trunk clearance.
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
