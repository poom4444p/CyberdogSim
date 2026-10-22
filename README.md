# Cyberdog Blind Navigation

A robotic guide dog: a navigation stack that takes a spoken command
("take me to room 201"), plans a route across a three-storey building, and walks
a quadruped there — announcing turns, taking the lift (the stairs only if the
person agrees), stepping around obstacles that are not on any map, and stopping
to let people past.

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
| [6](#6-the-vlm-server) | Start the VLM server | for the default run; skip it with `--no-vamos` |
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
python -m cyberdog.sim.run_building "room 201"              # needs the VLM server (step 6)
python -m cyberdog.sim.run_building "room 201" --no-vamos   # map route only, no server
```

That is the whole stack in one line: routing, the lift, voice announcements,
LiDAR, obstacle avoidance, VAMOS in the steering loop, and a video written to
`output/scene_cache/building.mp4`. Add `--no-video` for a faster run.

It runs the **release configuration** by default: VAMOS steers (`--vamos`),
because the real-world robot needs it, and the map route stays the fallback
whenever no VAMOS path passes the safety gate. When there are people, they
behave as an aware crowd that makes room for the dog (`--crowd aware`), and
the dog goes round them with CE-RRT\* instead of waiting (`--planner cerrt`).
This is the configuration Gate D passed on (`docs/dreaming_safety_kpis.md`,
2026-10-14 and 2026-10-20), in simulation, with base VAMOS.

VAMOS needs its server (step 6). Without one, the run stops at once with
`FAILED BY vamos: server is not answering`; start the server, or add
`--no-vamos` to drive the map route alone. The same applies to every command
below.

At the end it prints a score (this example is from a `--no-vamos` run; with
VAMOS the numbers differ a little, and extra `VAMOS floor` and `VLM ASYNC`
lines report its calls):

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

### 6. The VLM server

Needed for the default run, which puts the VAMOS vision-language model in the
steering loop. With `--no-vamos` the dog still navigates and avoids obstacles
on the map route alone, so you can skip this step for a first run. It runs in
a second environment, the same split the real robot has, where the VLM is a
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

To serve the LoRA adapter instead (`scripts/vamos/README.md`), pass it with
an absolute path -- from here `$PWD` is the server directory, not the repo:

```bash
bash start_server.sh --model_path /path/to/Cyberdog/models/vamos_lora
bash start_server.sh --model_path mateoguaman/vamos    # back to the base model
```

The server keeps whichever model it loaded last, including one loaded by
`evaluate.py --load`, until it is restarted. Restart it before every Gate D
campaign so the campaign is scored on the model you meant.

The first start downloads the `mateoguaman/vamos` weights (several GB), so
expect a long pause. From another terminal, check it is up, then run with it:

```bash
curl -s http://127.0.0.1:8009/health && echo " — server up"

conda activate cyberdog_sim
python -m cyberdog.sim.run_building "room 201"
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
`data/building/locations.json`, exactly or nearly: a near miss (`"libary"`,
`"chemistry lap"`) is matched, and the dog asks *"Did you mean the library?"*
and waits for continue before it moves. Name the place, or add `--nlu`.

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
Every `run_building` command steers with VAMOS and needs the VLM server
(step 6) unless it passes `--no-vamos`.

| Command | What it does |
|---|---|
| `python -m cyberdog.sim.run_building "room 201"` | Full stack, three floors, VAMOS steering, with video (needs the server) |
| `python -m cyberdog.sim.run_building "room 201" --no-video` | Same, faster — no rendering |
| `python -m cyberdog.sim.run_building "room 201" --no-vamos` | Map route only, no VLM server needed |
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3` | People walking the corridors, on no map — the dog goes round them (CE-RRT\*), and waits only where no path is safe |
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --crowd oblivious --planner map` | The stress test: people who only stop for a dog right in front of them, and a dog that waits for them instead of going round |
| `python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --seed 4` | Same, a different crowd (a seed is reproducible) |
| `python -m cyberdog.sim.run_building restroom --handle "tug@10,continue@15"` | The person on the handle: a tug at 10 s stops the dog until continue at 15 s; `pull@T1-T2` slows it, `push@T1-T2` undoes a pull. With `--handle`, a continue is also what answers the lift prompt, in place of `--auto-confirm` (`sim/handle.py`) |
| `python -m cyberdog.sim.run_building "take me upstairs" --nlu` | Parse the command through the Gemma layer (needs `.[language]`) |
| `python -m cyberdog.sim.run_building "room 201" --vamos --vlm-latency 1.8` | VAMOS in the loop, repeatable tick for tick (fixed answer latency) |
| `python -m cyberdog.sim.run_building "room 201" --lift-out-of-service` | Floor changes by the stairs, after asking twice (`--no-stairs`: never) |
| `python -m cyberdog.sim.run_building "room 201" --speed 4` | Play the video back 4× faster (control still runs at 20 Hz) |
| `python -m cyberdog.sim.scene.build_scene --building` | Rebuild all three storeys |
| `python -m cyberdog.sim.scene.build_scene 2` | Rebuild one floor only |
| `bash scripts/check.sh` | `pytest`, then the self-test, stopping at the first failure (`gate all` adds the full Gate D campaign, ~50 min) |
| `python scripts/gate_d.py --seeds 1 2 3` | Gate D: map-only vs `--vamos`, paired run by run (needs the server; `tests/README.md`) |
| `python scripts/planner_ab.py output/gate_d/<without> output/gate_d/<with>` | The planner's gate: two Gate D campaigns, `--planner map` vs `cerrt` |

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
| [`planning/`](src/cyberdog/planning/README.md) | Destination → route across floors, including lift or stairs and which refusals. Then projects the next checkpoint into a camera pixel for the VLM. CE-RRT\* plans the way round people. | `building_router.py`, `checkpoint_projector.py`, `ce_rrt.py` |
| [`sim/`](src/cyberdog/sim/README.md) | The twin: the Go2, a virtual Mid-360 LiDAR, perception and tracking, imagined rollouts, the person on the handle, and VAMOS in the steering loop. | `run_building.py`, `sensing/` |

`"map decides WHERE, VLM decides HOW"` is the division of labour between
`planning/` and the VLM, and it is worth knowing before reading either.

---

## Layout

```
src/cyberdog/
  paths.py            every filesystem location, in one place
  main_planner.py     command -> route, no simulator (needs [language])
  language/           the command parser
  mapping/            grids, zones, A*, the keep-right lane
  planning/           multi-floor routing, pixel projection, CE-RRT*
  affordance/         the Isaac-trained walkability network (shadow mode)
  sim/
    control.py        the control law and its gains
    clearance.py      static clearance from the map (walls + no-go zones)
    handle.py         the Smart Handle mock and the safety mux
    autolabel.py      walked paths -> VAMOS training pairs
    overlay.py        drawing onto the camera frame
    vamos_client.py   the VLM as a service: 5 candidates -> 1 chosen
    scene/            building the MuJoCo XML
    robot/            the Go2: kinematics, gait, pose
    sensing/          lidar -> perception/tracking -> dreaming
    run_building.py   ENTRY POINT: full stack, three floors
    run_demo.py       one floor, headless
    record_demo.py    one floor, with video

config/               map_config.yaml, camera_config.yaml
data/building/        grids, meshes, locations.json, zones.json
docs/                 spec.md, architecture.md, dreaming_safety_kpis.md (the gates),
                      known_limits.md (what was fixed, what is still open)
scripts/              check.sh, gate_d.py, planner_ab.py, ce_rrt_calibrate.py, train_affordance.py
scripts/mapping/      rebuilding grids and meshes from point clouds
scripts/language/     interactive parser prompt
scripts/vamos/        the VAMOS LoRA: collect, train, evaluate (deferred to hardware)
scripts/isaac/        the affordance data collector for Isaac Lab
tests/                unit tests + selftest.py (the layer-by-layer harness)
vendor/VAMOS/         the upstream VAMOS repo and its VLM server

models/   datasets/   output/      generated or trained — all gitignored
```

Nothing the repo tracks is ever written into the source tree: built scenes and
recorded video go to `output/`, weights to `models/`, generated datasets to
`datasets/`.

---

## What is and isn't verified

Everything below is **in simulation only**. The detailed history (what each
fix was, and the tests that used to pass while hiding a bug) is in
[`docs/known_limits.md`](docs/known_limits.md).

**The gates (2026-10-22).** Defined in
[`docs/dreaming_safety_kpis.md`](docs/dreaming_safety_kpis.md):

| Gate | What it asks | Status |
|---|---|---|
| A | A learned dream beats the kinematic one | not run: there is no learned dream yet (after hardware) |
| B | The safety factor is discriminating, calibrated, stable | B1–B17 pass (`selftest.py dreaming`) |
| C | VAMOS in shadow changes no command | pass, fake and real server |
| D | `--vamos` no worse than map-only, 70 paired runs | pass, aware crowd, base VAMOS (2026-10-20, `output/gate_d/20261020-172845`); narrowly on time: +6.8%, bootstrap 95% CI +3.6% to +10.4% |
| Planner | CE-RRT\* no worse than waiting, per arm | P2–P6 pass on both arms |

**VAMOS is always the default** (since 2026-10-21), because the real robot
steers with it; `--no-vamos` is the map-only baseline. What passed is *base*
VAMOS, zero-shot. A LoRA fine-tune on the twin's frames failed Gate D (D2, D3:
it learnt to draw straight lines to the goal), so fine-tuning is done in the
hardware phase, on real camera frames from the real building.

**Verified against ground truth** the robot is never shown:

- Every named destination on all three floors arrives with **zero
  collisions**, and the self-test's 8 routes with no contact for the person
  on the handle (`selftest.py`, `person on the handle untouched`).
- Floors change by the lift. The stairs are a fallback (lift out of service,
  or much further), taken only after the person agrees twice: before setting
  off, and again at the top step. Asked for the stairs as a destination, the
  dog stops in the corridor beside them and says so.
- Crates that are not on the map are found by LiDAR with zero false positives
  (`selftest.py`, `no false positives`).
- Each run that does not arrive says which layer ended it (`FAILED BY
  <layer>`, `STOPPED BY handle`, `REFUSED BY language|planning`).
- Near-miss place names ("libary", "servar rum") are matched by spelling or
  sound and confirmed on the handle before the dog moves; a no is
  `STOPPED BY language`.
- Latency on an M-series Mac is within budget (VAMOS call 1.6–1.8 s against
  2–3 s; table in [`docs/architecture.md`](docs/architecture.md#timing-budget)).
  Not yet measured on a Jetson.

**Known limits** (details in [`docs/known_limits.md`](docs/known_limits.md)):

- Gate D's time margin is narrow: `cafeteria` (+15%) and `room 201` (+18%)
  are the slow routes.
- Under 1.5× sensor noise some passed paths collide often (B11), and paths
  scoring near the 0.5 gate can flip between seeds (B14). Both are report-only;
  `selftest.py dreaming` prints the counts.
- People stand still inside the dream (it reads the last LiDAR scan), and
  the pedestrians are scripted capsules, not a pedestrian model.
- CE-RRT\* still stops where no path keeps P(contact) < 1%, which in a 2.5 m
  corridor is often.
- Perception has little memory (a crate is remembered 6 s, `MEMORY_S`; a
  person who steps behind a crate is a lost track), and an obstacle against
  a wall is invisible.

Still to do for the spec's Definition of Done: latency and success metrics
recorded, and the go/no-go report ([`docs/spec.md`](docs/spec.md) §7).
