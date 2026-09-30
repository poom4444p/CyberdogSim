# Tests

How to run every test in this folder, what each one checks, and what it needs.
Run everything from the **repo root** in the `cyberdog_sim` environment (not
`vamos_mac`):

```bash
conda activate cyberdog_sim
```

| What | Command | Time | Needs |
|---|---|---|---|
| Unit tests | `pytest` | ~1 s | nothing |
| Self-test, every stage | `python tests/selftest.py` | ~1.5 min | the building scene |
| Self-test, one stage | `python tests/selftest.py lidar` | 1–40 s | the building scene |
| Location check | `python tests/test_locations.py` | ~2 s | nothing |
| Location check with the parser | `python tests/test_locations.py --with-model` | slow | the trained parser |

Before a commit, run `pytest` and `python tests/selftest.py`. Both should end
clean (`107 passed`, `ALL PASS`).

## Before the first run

1. Install the package and the Go2 model -- step 1 of the
   [main README](../README.md#1-install). `MENAGERIE_PATH` must be set if the
   menagerie is not at `~/mujoco_menagerie`.
2. Build the building scene once. The self-test needs it; the unit tests do not:

   ```bash
   python -m cyberdog.sim.scene.build_scene --building
   ```

   Rebuild it after changing the map, the lift or the obstacles.

---

## 1. Unit tests (`pytest`)

Fast, no simulator, no scene. `pytest` runs every `test_*.py` file here and
skips `selftest.py` on purpose (`pyproject.toml`).

```bash
pytest                                  # all of them
pytest tests/test_astar.py              # one file
pytest tests/test_astar.py -k corner    # tests whose name matches
pytest -q                               # quieter
```

| File | Checks | Tests |
|---|---|---|
| `test_occupancy_grid.py` | The occupancy grid: construction, world ↔ grid coordinates, editing cells, save/load | 14 |
| `test_astar.py` | The A\* planner: search, path simplification, checkpoints, config | 24 |
| `test_behavior_layer.py` | Zones: point-in-zone tests, queries where zones overlap and what action wins, save/load | 22 |
| `test_router_approach.py` | Where the dog stands when asked for a no-go place like the stairs, on every floor | 5 |
| `test_overlay_labels.py` | Room labels in the video: projected where the camera sees them, never overlapping | 16 |
| `test_dataset.py` | The command parser's training vocabulary (no model needed) | 14 |

## 2. Location check (`test_locations.py`)

A script, not a pytest file. It checks that the building matches what the
command parser can say:

```bash
python tests/test_locations.py                 # map only
python tests/test_locations.py --with-model    # also runs the trained parser
```

- Every place the parser can output exists in `data/building/locations.json`.
- Every place is on free floor.
- There is a route from the main entrance to each one, and the stairs are
  refused as unsafe.

A good run ends with `Map check: 45/45 locations OK` and `All good.`
`--with-model` needs the trained parser (main README, step 2) and the
`[language]` extras.

## 3. Self-test (`selftest.py`)

The whole twin, one layer at a time, so a failure names the layer that broke.
Every check prints `[PASS]` or `[FAIL]` with the number it judged on, and the
run ends with `ALL PASS` or a list of what failed.

```bash
python tests/selftest.py                  # every stage, bottom-up
python tests/selftest.py run              # just one
python tests/selftest.py lidar crowd      # several
```

Stages run bottom-up. **Fix the first failure first**: a broken scene fails
every stage above it.

| Stage | Checks | Time |
|---|---|---|
| `scene` | The building and the dog are the right size, and the walls are where the map says | ~1 s |
| `lidar` | The sensor sees a crate that is not on the map, fast enough for a 20 Hz loop | ~1 s |
| `perception` | Crates are flagged as obstacles, and an empty corridor is not | ~2 s |
| `destination` | A goal inside a crate is moved beside it, and a clear goal is left alone | ~2 s |
| `latency` | What the sensing layers cost per control tick | ~2 s |
| `crowd` | A person is told apart from a crate, and the dog stops for someone coming towards it | ~2 s |
| `dreaming` | The path safety score (Gate B of `docs/dreaming_safety_kpis.md`): safe vs unsafe, calibrated against real drives, stable across seeds | ~40 s |
| `vamos` | The VAMOS client against a **fake** server, and shadow mode (Gate C): steering and shadow runs give identical commands | ~20 s |
| `run` | The whole stack on 8 routes across all 3 floors. Each must arrive with no collision, no pedestrian contact, and **no contact for the person on the handle** | ~25 s |

The `vamos` stage runs its own stand-in server. It needs neither the real
VLM server nor `vamos_mac`.

To run without switching environments:

```bash
conda run -n cyberdog_sim --no-capture-output python tests/selftest.py
```

---

## 4. Checks by hand

These are not in any test file yet. Use them when a change touches steering,
avoidance or people.

### One route, with its full report

```bash
python -m cyberdog.sim.run_building "room 201" --auto-confirm --no-video
```

The last lines are the verdict. Lines in capitals are problems:

| Line | Means |
|---|---|
| `ARRIVED` / `FAILED` | Did it get there |
| `COLLISIONS:` | The dog's body touched a wall or an obstacle (`collisions: none` if clean) |
| `HANDLER:` | The person on the handle touched something (`handler: none` if clean) |
| `CONTACT:` | Came within 0.55 m of a pedestrian (`contact: none` if clean) |
| `commands: … sha256 …` | Fingerprint of every command sent; two runs match only if they drove identically |

Drop `--no-video` to get an MP4 (`--out path.mp4`). The left panel films the
dog from the front, so it is mirrored: the dog's right is on the screen's left.

### Pedestrian sweep

People make runs sensitive: one small change moves the whole run onto a
different path. Judge a change on many seeds, not one:

```bash
for s in $(seq 1 20); do
  echo "seed $s: $(python -m cyberdog.sim.run_building 'room 201' --pedestrians 3 --seed $s \
      --auto-confirm --no-video 2>&1 | grep -E '^(FAILED|COLLISIONS|HANDLER|CONTACT)' | tr '\n' ' ')"
done
```

A seed that prints nothing arrived clean. Measured at commit `859bbf5`: all 20
arrive with no dog collisions, 3 have handler contact, 8 have pedestrian
contact (8.4 s in total).

### With VAMOS

Start the real server first (main README, step 6), then:

```bash
python -m cyberdog.sim.run_building "second floor restroom" --nlu --vamos \
    --pedestrians 4 --seed 2 --auto-confirm --no-video --vlm-latency 1.8
```

Always pass `--vlm-latency` when comparing runs. With it, the same command
repeats tick for tick (same `commands:` sha256). Without it, answers land after
the server's measured response time, which varies, and runs drift.
`--vamos-sample` makes VAMOS sample randomly again. Don't use it when comparing.

### Gate D (VAMOS vs map-only)

The promotion test in `docs/dreaming_safety_kpis.md`: every route × at least
3 pedestrian seeds, once without and once with `--vamos`. Both runs of a pair
use the same command and seed:

```bash
python -m cyberdog.sim.run_building "room 101" --pedestrians 3 --seed 1 --auto-confirm --no-video
python -m cyberdog.sim.run_building "room 101" --pedestrians 3 --seed 1 --auto-confirm --no-video \
    --vamos --vlm-latency 1.8
```

Routes: `restroom`, `cafeteria`, `room 201`, `room 101`,
`electrical engineering lab`, `biology lab`, `chemistry lab`. That is 42 runs
for 3 seeds. Add up collisions, handler and pedestrian contact, arrivals and
time for each arm. The thresholds are in the KPI doc (D1–D8). There is no
script for this yet.

---

## When something fails

- **`no building scene yet`** or **`no scene yet`** -- build it (see [Before the first run](#before-the-first-run)).
- **The Go2 model cannot be found** -- `MENAGERIE_PATH` is not set in this
  shell. Set it, then `conda activate cyberdog_sim` again.
- **`ModuleNotFoundError: cyberdog` or `mujoco`** -- wrong environment. Run
  `conda activate cyberdog_sim`, or `pip install -e ".[language,dev]"` if it
  was never installed.
- **`VAMOS server is not answering`** -- only `--vamos` / `--shadow` runs need
  it. Start it (main README, step 6) or drop the flag.
- **A `run` route fails** -- rerun that one route by hand (section 4) to see
  its full report, then run the lower stages to find which layer moved.
