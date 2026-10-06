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
clean (`193 passed`, `ALL PASS`).

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
| `test_handle.py` | The Smart Handle mock and the safety mux: a tug stops and latches until continue, a pull lowers the pace, a push only undoes a pull, and no force leaves every command unchanged | 25 |
| `test_affordance_data.py` | The affordance data contract shared with the Isaac Lab collector: the elevation patch (frame, highest-wins, NaN for unseen), the ground under the dog, the shard format; and the labels -- walkable / caution / not walkable, bumps measured on the path only, a ramp is a slope not a bump, the gait's own rocking is not a failure | 27 |
| `test_affordance_model.py` | The affordance network's plumbing (needs torch): input features and the jump map (an unseen border is not an edge), left-right mirroring, the 0.2 refusal rule, save/load | 10 |
| `test_affordance_runtime.py` | The affordance network on a running dog: the elevation memory (the ceiling is not ground, a person who walked on leaves no ghost, what was not seen again is remembered, it forgets after 3 m, a new floor starts empty) and the walk-ahead target in the dog's frame | 10 |
| `test_router_stairs.py` | Lift or stairs for a floor change: the lift unless out of service or 2x and 30 m further; the stair waiting point is outside the stairwell; walking routes never cross one | 14 |

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
| `affordance` | The affordance network (needs `models/affordance/mlp.pt`) walked up to every crate on the map with real scans: no walk into a box is passed. Reports, without failing, walks whose body only touches at the end and clear walks beside a box that it refuses | ~6 s |
| `handle` | The person on the handle overrules the dog: a tug stops it where it stands until continue (walking and boarding the lift), with no continue the run ends there, the lift waits at its doors for a continue of its own, and an idle handle changes no command | ~5 s |
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

A seed that prints nothing arrived clean. Use more seeds than you think: on
20, one change looked like a regression that 40 showed was one seed. Change
`seq 1 20` to `seq 1 40` above. Measured over 40 seeds after the box-pinch
fixes: all arrive, no dog collisions, 6 seeds with handle contact (5.8 s, all
walls), 7 with pedestrian contact (8.5 s).

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
3 pedestrian seeds, once without and once with `--vamos` (same command, same
crowd), scored on D1–D8. Start the VAMOS server first, then:

```bash
python scripts/gate_d.py                                   # 7 routes x seeds 1 2 3 = 42 runs, ~20 min
python scripts/gate_d.py --seeds 1 2 3 4 5                 # more seeds
python scripts/gate_d.py --routes "room 101" --seeds 1     # one pair, a quick check
python scripts/gate_d.py --resume output/gate_d/<time>     # finish a campaign that stopped
```

The VAMOS server has died part-way through long campaigns (after 6–21
candidate runs). When that happens the script says so and prints the
`--resume` command: restart the server and run it. It keeps every run that
finished cleanly (a candidate run only if VAMOS answered every call) and
drives the rest into the same directory, with the same routes and seeds.

It prints one line per run as it finishes, then the D1–D8 table and
`Gate D: PASS` or `FAIL` (exit code 0 or 1). Every run's full output is kept
under `output/gate_d/<time>/`, with `summary.json` beside it. The candidate
runs always use `--vlm-latency 1.8`, so a rerun gives the same numbers. A
candidate run in which VAMOS was never called, or a call failed, counts
against D8: it would otherwise pass by quietly being the baseline.

One pair by hand, to look at a single result:

```bash
python -m cyberdog.sim.run_building "room 101" --pedestrians 3 --seed 1 --auto-confirm --no-video
python -m cyberdog.sim.run_building "room 101" --pedestrians 3 --seed 1 --auto-confirm --no-video \
    --vamos --vlm-latency 1.8
```

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
