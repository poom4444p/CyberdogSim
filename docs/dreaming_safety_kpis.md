# Dreaming Safety Gates — KPI Checklist

What has to be true before anything that *imagines* the future — today the
safety factor in `sim/sensing/dreaming.py`, later whatever replaces it — is
allowed to steer the dog.

Adapted from the Dreamer promotion gates in two related projects by Moha-C:

- https://github.com/Moha-C/autonomous-vehicule-VLA — the public release; the
  source of the shadow-trace, paired A/B and promotion gates (C, D) and the
  online-update rollback rules (E)
- https://github.com/Moha-C/vla-av — the research repo; the same C/D scripts,
  plus the world-model-vs-persistence gate (A) in `src/residual_dreamerv3/`

The structure is theirs; the thresholds and test cases below are CyberDog's.
One deliberate difference: both repos split train/test by seed, and Gate A here
splits by floor and layout, so no layout is seen in both.

## What is being gated

Both projects gate a *learned* world model. CyberDog's dream is not learned: it is
the live loop's own pure-pursuit law (`sim/control.py`) rolled forward 24 times
with noise on heading bias, speed and slip, returning

    safety = P(no collision) x (room left)

and it gates (`GATE = 0.5`, `vamos_client.py`), ranks and throttles VAMOS
candidates. Nothing is trained, so nothing can overfit or drift — which is why
two of their gates do not apply yet:

| Gate | Question | Applies |
|---|---|---|
| A | Does a learned model predict better than a trivial baseline? | **Later** — when the spec's upgrade path (spec Layer 6, item 3) swaps the kinematic rollout for a learned locomotion model or the affordance MLP |
| B | Does the dream tell a safe path from an unsafe one, and is it right? | **Now** |
| C | Shadow mode: can the dream run without touching the controls? | **Now** |
| D | Paired A/B: does steering with it actually do better? | **Now** — the promotion gate |
| E | Online-update rollback | **No** — CyberDog does not learn online |

Gates run in order **B → C → D**, and **A** is inserted before B the day the
dream becomes a model. Never let a new scorer steer until B passes: a scorer
that is wrong about the world can look decisive while making the dog less safe.

Design principle, as everywhere else: *AI proposes, simple code disposes.* The
dream is a proposer's filter; the deterministic map route stays the fallback
and none of these gates is allowed to weaken it.

---

## Gate A — A learned dream beats the kinematic one (offline, future)

Not run today. Written down now so the upgrade has a bar to clear.

| ID | KPI | Threshold | Notes |
|---|---|---|---|
| A1 | **Gain vs the kinematic dream**: `(err_kin − err_model) / err_kin`, multi-step pose error against the MuJoCo twin, each dim divided by its train std, averaged over horizons 0.5 / 1 / 2 / 5 s | **≥ +2%** | the kinematic dream is the baseline to beat, not persistence — it already exists and is already good |
| A2 | Gain vs persistence (dog keeps its current velocity) | report only | sanity floor |
| A3 | Discrete channels (foot contact, gait phase, collided) scored as accuracy, not MSE | report only | |
| A4 | Evaluated on **unseen floors / layouts**: split by floor and obstacle layout, never by seed | mandatory | train on floors 1–2, test on 3, and rotate |

- [ ] A1 passes on the held-out floor
- [ ] A1 reported with 95% CI
- [ ] Split is grouped — no layout appears in both train and test

---

## Gate B — The safety factor is discriminating, calibrated and stable

Where: `python tests/selftest.py dreaming`. Status column is today's.

### B.1 Discrimination (regression cases, floor 1)

| ID | Case | Threshold | Status |
|---|---|---|---|
| B1 | Down the middle of the corridor | factor ≥ 0.9 | ✅ checked |
| B2 | Through the room door at x = 9 | `GATE ≤ factor < corridor` | ✅ checked |
| B3 | 0.5 m off the south wall | factor < `GATE` | ✅ checked |
| B4 | Into a wall / into the stairwell | factor **= 0.00** exactly | ✅ checked |
| B5 | **Stopping lowers risk**: a risky path cut off half-way vs the whole of it, same seed | mean factor advantage **≥ 0.005** | ✅ checked |
| B6 | **Stopping never raises collision**: P(safe) of the half path minus the whole | mean **≥ 0** (per path is binomial noise at 24 rollouts) | ✅ checked |

B4 is the spec's acceptance line: a candidate aimed at a stairwell scores 0.00
without anything stair-specific in the scorer.

### B.2 Gate accuracy (dreamed vs driven on the twin)

Candidates shaped like VAMOS's — 2–3 m, fanning sideways, into doors — each
dreamed, then driven on the twin with the live loop's `MAX_V` / `MAX_W`.

| ID | KPI | Threshold | Status |
|---|---|---|---|
| B7 | Candidates measured | **≥ 128** | ✅ checked |
| B8 | Hazard candidates (line passes closer than `ROOMY` = 0.40 to a wall, door jamb or the stairwell, in clearance-field units) | **≥ 32** | ✅ checked |
| B9 | **False passes**: passed candidates whose driven P(collision) > 0.2 — screened at 30 drives, confirmed at 200 | **0** | ✅ checked |
| B10 | Calibration: mean \|dreamed P(safe) − driven P(safe)\| | **≤ 0.10** | ✅ checked |
| B11 | Stress: false passes with the twin's noise ×1.5 | report only | ✅ printed |
| B12 | Rejected though every drive was clean | report only (by design in tight rooms) | ✅ printed |

Read B9–B10 for what they are: the twin is the same kinematic model the dream
imagines, so this catches the dream disagreeing with the robot it claims to
model, not the model disagreeing with the world. Gate A is where that second
question gets asked.

### B.3 Stability and sanity

| ID | KPI | Threshold | Status |
|---|---|---|---|
| B13 | Clear-cut paths (corridor, into wall) flip verdict across 20 seeds | **0 flips** | ✅ checked |
| B14 | Near-gate paths: mean ± sd and flips over 20 seeds, and driven P(collision) of what a flip lets through | report only | ✅ printed |
| B15 | **Collapsed output**: hazard frames where the fanned candidates all get the same factor below 1.00 (tol 1e-4); all-1.00 frames are counted apart as saturated | **≤ 25%** of frames | ✅ checked |
| B16 | Every factor and P(safe) finite and in [0, 1], on ≥ 2 seeds | mandatory | ✅ checked |
| B17 | Five candidates scored inside one control tick | **< 50 ms** | ✅ checked |

- [x] B1–B17 all pass
- [x] Candidate count raised to ≥ 128 with ≥ 32 hazards

Fixed after B15 surfaced it: the room term was the closest approach over the
whole walk, so from a pose already near a wall it read the shared start and
every candidate tied — steering away from the wall earned nothing (7 of 50
hazard frames). `dreaming.START_R` now counts only closing in within 0.5 m of
the start; ties are down to 1 of 50.

---

## Gate C — Shadow mode

The dream runs on every VAMOS call, but the dog is driven by the map route
alone. Proves the scorer can be switched on without changing what the dog does.

Where: `python tests/selftest.py vamos`, against a stand-in VAMOS server
(five lines fanned at the goal pixel), so it needs nothing else running. By
hand: `run_building <route> --shadow [--shadow-log PATH] [--vamos-url URL]`.
Every run prints `commands: N ticks, sha256 …`, a fingerprint of every
velocity command sent; equal fingerprints mean identical commands.

| ID | KPI | Threshold | Status |
|---|---|---|---|
| C0 | Map-only run repeats itself (same fingerprint twice) — without this C1 proves nothing | bit-for-bit | ✅ checked |
| C1 | Commanded `(vx, vy, wz)` identical to a map-only run of the same route and seed | bit-for-bit, **100% of ticks** | ✅ checked (room 101, fake and real server) |
| C2 | Shadow log written per call: candidates, factors, which one *would* have been chosen | present for every call | ✅ checked |
| C3 | Shadow disagreements: calls where the dream would have steered > 15° off the map route (`steer`), or had nothing pass the gate (`none`) | report only | ✅ printed |

C3 is the interesting number: it is what Gate D is about to test. With the
fake server it says nothing about VAMOS — run `--shadow` against the real
server for that.

The same stage also checks the client's disposing in `plan()`: a line through
a wall is thrown out as off-map, one hugging a wall by the gate, the clear one
is chosen, and with nothing surviving or the server down it returns no path.

- [x] Shadow trace verified: the dream never changed a command (fake server)
- [x] Same, with the real VAMOS server

Real-server run, 2026-09-29, `room 101`, commit `7f422db`, VAMOS on `mps`,
answers arriving asynchronously at their measured latency:

| | Map-only | Shadow (real VAMOS) |
|---|---|---|
| Commands | 1270 ticks, sha256 `71dec15b85f45b7d` | 1270 ticks, sha256 `71dec15b85f45b7d` |
| Outcome | ARRIVED, 66 s, no collisions | ARRIVED, 66 s, no collisions |
| VAMOS answers | — | 29 (30 asked, 1 dropped at the leg's end), 1.6 s old on arrival |
| Candidates rejected | — | 99, of which 16 by the gate (the rest off-map or behind the dog) |
| C3 `steer` | — | 2 of 29 calls |
| C3 `none` | — | 17 of 29 calls |

Log: `output/shadow_real_room101.jsonl`. One run of a sampling model at
`temperature=1.0`, so C3 is indicative, not a measurement: an earlier
real-server shadow run of the same route had 0 `steer` and 64 `none` in 90
calls. On a route the map already solves, real VAMOS mostly offers nothing
that passes, and rarely anything different.

---

## Gate D — Paired closed-loop A/B (promotion)

Baseline: map-only (`run_building` without `--vamos`). Candidate: `--vamos`
with the dream gating. The README already records `--vamos` scoring *worse*
than map-only on floors the map solves — this gate is what stops that being
shipped.

| ID | KPI | Threshold |
|---|---|---|
| D1 | Paired runs: the 8 routes of `selftest.py run` × **≥ 3** pedestrian seeds (`--pedestrians 3 --seed N`), both arms, equal counts, campaign complete | mandatory |
| D2 | **Collisions total** (crate / wall footprint entries) | candidate ≤ baseline |
| D3 | **Pedestrian contacts total** (inside 0.55 m) | candidate ≤ baseline |
| D4 | **Stair or no-go zone entries** | **0** in both arms — any entry fails outright |
| D5 | **Arrival rate** (`ARRIVED` on the intended floor) | candidate ≥ baseline |
| D6 | Time to goal, mean over arrived runs | candidate ≤ baseline × 1.10 |
| D7 | Minimum clearance to crates and people | report, mean and worst |
| D8 | Missing metrics | **0** — a missing collision line is not zero collisions |

D2 counts MuJoCo contacts between the whole Go2 body (trunk, hips, legs)
and the walls, crates, lift and stairs, so a graze counts. It replaced a
point test on the dog's centre against the crates only. Its first finding,
map-only scraping the south wall on three routes, is fixed: routes keep
right and no longer cut door frames, and the map-only baseline is clean on
all 8 routes.

D2 also counts the person on the handle (`HANDLER:`): a guide that keeps
its own body clear while swinging its user into a door frame has not
passed. The map-only baseline is clean on all 8 routes (down from 7 of 8),
and `selftest.py run` now fails on any contact. With pedestrians it is not
yet zero: 2 of 12 seeds of `room 201 --pedestrians 3` (README, known limits).

Run it with `python scripts/gate_d.py` (7 routes, VAMOS by beam search at
`--vlm-latency 1.8`), or `bash scripts/check.sh gate all` for seeds 1–10
after pytest and selftest. First campaigns, 2026-09-30, seeds 1–3:

| KPI | map-only | `--vamos`, first run | `--vamos`, after the fixes below | |
|---|---|---|---|---|
| D1 pairs complete | 21 | 21 | 21 | PASS |
| D2 collisions + handle (s) | 0.0 | 1.7 | 0.5 | FAIL |
| D3 pedestrian contact (s) | 2.0 | 2.8 | 2.9 | FAIL |
| D4 stair / no-go entries | 0 | 0 | 0 | PASS |
| D5 arrivals | 21/21 | 21/21 | 21/21 | PASS |
| D6 mean time (s) | 53.3 | 65.8 (+23%) | 55.9 (+5%) | FAIL → PASS |
| D8 missing / VAMOS unused | – | 0 / 0 | 0 / 0 | PASS |

Logs: `output/gate_d/20260930-141041` (first) and `…-145515` (after).
The fixes:

- **No crawling because VAMOS had nothing.** The dog slowed to 30% whenever
  no VAMOS path passed and the LiDAR saw anything, even with the route
  clear: 266.7 s of crawling over the 21 candidate runs, 0 s map-only. It
  now crawls only when the map has no way past either.
- **The route steers into doors** (within 4 m, `DOOR_MAP_D`). A VAMOS path
  kept the dog in its lane past the point the route crosses for a door, and
  the turn left at the door overshot: chemistry lab +32 s, now +0.
- **The dream counts the person on the handle** (`dreaming.PERSON_R`). A
  VAMOS path that swings them into something is a colliding rollout. The
  first run's 1.6 s against the floor-1 trolley (`room 101` seed 1) is gone.

What is left is one pair each. D2: `room 201` seed 1, 0.5 s of the person
against the floor-2 cart, squeezing past it after a pedestrian stepped into
the gap. D3: the same run (1.5 s) and `room 101` seed 3 (1.4 s, against the
baseline's 2.0 s on that pair). The cart-and-pedestrian pinch shows up in
map-only runs on other seeds too, so it is a crowd problem more than a
VAMOS one -- but the gate is candidate ≤ baseline, and it is not.

The VAMOS server stopped mid-campaign twice (`gate_d.py` now detects it
and reports the gate incomplete). With free memory down to 0.1 GB while it
and four map-only runs share the machine, memory is the likely cause; the
campaign that completed ran with the server's output logged, and it did not
fail that time.

### 2026-10-08: 7 routes × seeds 1–10 (70 pairs)

Each column is one campaign, map-only / `--vamos`:

| KPI | `main` | + sidestep | + sidestep + stopped person | |
|---|---|---|---|---|
| D1 pairs complete | 70 / 70 | 70 / 70 | 70 / 70 | PASS |
| D2 collisions + handle (s) | 1.9 / 6.9 | 1.4 / 2.9 | 2.9 / 2.7 | FAIL → PASS |
| D3 pedestrian contact (s) | 4.0 / 6.6 | 3.9 / 4.8 | 4.7 / 3.6 | FAIL → PASS |
| D4 stair / no-go entries | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| D5 arrivals | 70/70 both | 70/70 both | 70/70 both | PASS |
| D6 mean time (s) | 54.2 / 58.2 | 54.4 / 57.7 | 56.6 / 58.9 | PASS |
| D8 missing / VAMOS unused | 0 / 0 | 0 / 0 | 0 / 0 | PASS |

Logs: `output/gate_d/20261008-094731`, `…-104538`, `…-120028`. An earlier
70-pair campaign (`20261007-161746`) ran with the stopped-person change
uncommitted in the tree and is not a `main` baseline. The fixes:

- **Step sideways when no eased turn is clear** (`SIDE_V`). Setting off
  round somebody who stopped in its lane, the dog had them 0.55 m in front
  and its user 0.5 m from the wall. Every eased turn failed the 0.7 m
  people check, stepping away included, so the full pivot went ahead and
  swung the user into the wall: `room 101` seeds 2, 3, 8, 9 and `room 201`
  seed 10, `--vamos`. Someone already inside `SWING_PEOPLE` now only has to
  get no closer.
- **A person who stopped stays a person** (`tracking.CLAIM_R`,
  `PASS_SIDE`). A walker who stopped beside a crate merged into one cluster
  with it and was passed at a crate's distance, 0.40 m (`room 201`).

The pass is narrow. Map-only handle contact rose 1.4 → 2.9 s with the
second fix, mostly `room 101` seed 3 (1.8 s, both arms touching there),
and `--vamos` clears D2 by 0.2 s. Not zero in either arm: the crate in the
`electrical engineering lab` (`--vamos` seeds 1 and 4, 1.4 s, a right turn
after passing it), `room 101` seed 3, and the lift on `restroom` seed 7
(0.3 s, both arms).

### 2026-10-11: the VAMOS LoRA adapter, seeds 1–10 (70 pairs)

The adapter of `scripts/vamos/README.md` (epoch 3, val loss 0.54) against
base VAMOS on the same code and seeds. The map-only arm repeats the
2026-10-08 one run for run.

| KPI | map-only | base VAMOS (`20261008-120028`) | LoRA (`20261011-200051`) |
|---|---|---|---|
| D2 collisions + handle (s) | 2.9 | 2.7 PASS | 3.3 **FAIL** |
| D3 pedestrian contact (s) | 4.7 | 3.6 PASS | 6.4 **FAIL** |
| D6 mean time (s) | 56.6 | 58.9 | 57.6 |

D1, D4, D5, D8 pass. In the loop it walks closer to people: `room 201`
seeds 9 and 10 (closest person 0.72 → 0.47 m and 0.70 → 0.43 m; 1.9 s and
1.2 s of new pedestrian contact, plus 1.0 s handle on seed 9), `room 101`
seed 3 (handle 0.7 → 1.7 s) and seed 8 (0.2 s). It does clear the
`electrical engineering lab` crate (seeds 1 and 4, 1.4 s → 0).

**It draws a straight line to the prompt's goal.** `room 201` seed 9 under
`--shadow`, base then adapter, walks the same 1337 ticks, so all 33 calls
see the same frames. The adapter's paths bend from straight by a median
2 mm (base 46 mm, worst 0.18 m against 0.73 m), are 2.44 m long almost every
call, and point within a median 1.7° of base's. No call offered a detour.
The labels' prompt is hindsight, the end of the walked path, so the end is
in the prompt and copying it scores well. Scored with the live prompt
(`map_prompt`, `*_map.json`), the end error does not move (0.231 base, 0.230
adapter); the 0.26 → 0.03 top-path drop is with the hindsight prompt. Fewer
of its candidates fall off the map or behind the dog (11% rejected against
28%), but the path it follows rates lower in the dream (0.86 against 0.93),
0.77 on floor 1 in 36 runs (base: 1.00 in 36).

The server keeps the last model it loaded. `20261009-091038` matches this
campaign run for run, so it was the adapter too, not base. The two seed
11–20 campaigns of 2026-10-09 match each other but did not record their
model; neither is a baseline. `summary.json` now keeps `model`.

**Deferred (2026-10-14).** No further fine-tuning in simulation: the twin's
frames are not the building's, so an adapter trained on them is tuned to the
twin. Gate D promotes base VAMOS; LoRA is retried on real frames in the
hardware phase (`docs/spec.md`, L4).

### 2026-10-14: aware crowd, base VAMOS, seeds 1–10 (70 pairs)

`--crowd aware`, `mateoguaman/vamos`. The campaign before this one failed D2
on a single run: `electrical engineering lab` seed 1, `--vamos`, 0.8 s of the
person on the handle against the floor-2 crate.

| KPI | map-only | `--vamos` | |
|---|---|---|---|
| D1 pairs complete | 70 | 70 | PASS |
| D2 collisions + handle (s) | 0.0 | 0.0 (was 0.8) | PASS |
| D3 pedestrian contact (s) | 1.7 | 1.2 | PASS |
| D4 stair / no-go entries | 0 | 0 | PASS |
| D5 arrivals | 69/70 | 70/70 | PASS |
| D6 mean time (s) | 54.8 | 59.3 | PASS |
| D7 closest person, mean / worst (m) | 2.80 / 0.13 | 2.80 / 0.25 | report |
| D8 missing / VAMOS unused | 0 | 0 / 0 | PASS |

Logs: `output/gate_d/20261014-113902`. The fixes:

- **The crate is remembered once it is behind the dog** (`perception.MEMORY_S`,
  `LiveClearance.remember`). The Mid-360 is pitched nose-down, so a 0.70 m
  crate 1 m behind is under its rays; the person on the handle reads what was
  seen standing in the last 6 s. Fixed seed 4.
- **An eased turn is held to no stricter a bar than the turn it replaces**
  (`spare_handler`, `need`). Seed 1 ruled out every eased turn and sidestep
  for coming within 0.15 m of the corridor wall's inflated band, and then made
  the pivot, which came as close to the same band and swung the person onto
  the crate. Only where the command would put them into something: applied
  everywhere, it stalled map-only seed 8 in the lab doorway for 37 s, where
  the band reads zero. As a side effect `cafeteria` seed 7 no longer spends
  207 s easing turns (253 s → 41 s map-only, 256 s → 42 s `--vamos`).

The one map-only miss (`room 101` seed 1, 173 s) predates these fixes.

- [x] D1–D8 all pass (2026-10-08, `output/gate_d/20261008-120028`)
- [x] D1–D8 all pass, aware crowd (2026-10-14, `output/gate_d/20261014-113902`)
- [x] Promotion is an explicit decision; never promote from selftest alone.
      **Decided 2026-10-15: not promoted.** `--vamos` stays opt-in. It passes
      Gate D, narrowly, but it is base VAMOS, zero-shot on the twin's frames;
      it is retried once tuned on real camera frames (hardware phase, spec L4)
- [x] **Decided 2026-10-21: base VAMOS is `run_building`'s default.** The
      real-world release steers with VAMOS, so the simulation now runs the
      configuration that will ship; `--no-vamos` is the map-only baseline, and
      `scripts/gate_d.py` passes it explicitly. The evidence is unchanged: Gate
      D passed in simulation only, narrowly on D6, with base VAMOS. Tuning on
      real camera frames is still pending (spec L4)

---

## The planner's gate — CE-RRT* against itself switched off

`--planner cerrt` (spec L6 s2) changes what the dog does about people in both
Gate D arms, so Gate D cannot judge it: with the planner on, the map-only arm
had no pedestrian contact at all, and Gate D then asks the `--vamos` arm for
none on all 70 runs -- one reshuffled encounter fails it. The planner is
judged the way Gate D judges VAMOS, against itself switched off, per arm,
paired run by run, on Gate D's bars (`scripts/planner_ab.py`):

| ID | KPI | Threshold |
|---|---|---|
| P2 | Collisions + handle contact | with ≤ without |
| P3 | Pedestrian contact | with ≤ without |
| P4 | Stair / no-go entries | 0 |
| P5 | Arrivals on the intended floor | with ≥ without |
| P6 | Mean time to goal | with ≤ without × 1.10 |
| P7 | Stops for people, waiting, gone round | report |

### 2026-10-14: aware crowd, base VAMOS, seeds 1–10

Without: `output/gate_d/20261014-113902`. With: `output/gate_d/20261014-202542`.

| KPI | map-only without → with | `--vamos` without → with | |
|---|---|---|---|
| P2 collisions + handle (s) | 0.0 → 0.0 | 0.0 → 0.0 | PASS |
| P3 pedestrian contact (s) | 1.7 → 0.0 | 1.2 → 0.8 | PASS |
| P4 stair / no-go | 0 → 0 | 0 → 0 | PASS |
| P5 arrivals | 69 → 70 | 70 → 70 | PASS |
| P6 mean time (s) | 54.8 → 57.2 | 59.3 → 61.4 | PASS |
| P7 stops / waiting (s) / gone round | 95 / 183 / 0 → 80 / 87 / 77 | 99 / 226 / 0 → 106 / 132 / 99 | report |

Gate D on the with-planner campaign itself: D3 FAILS, 0.8 s against 0.0 --
all of it `room 201` seed 10 `--vamos`, a walker cutting diagonally across
the corridor at a dog that had just waited for somebody else. Gate D for
promoting VAMOS stays the no-planner campaign above.

What it took (each found by a campaign or a 70-run map-only sweep, each fixed
before the next): the margin, measured instead of guessed -- the guess was a
third of the real error; a goal held for the whole going-round instead of
sliding with the dog (stop-start); a goal kept short of the next corner (a
plan that cut the cafeteria corner left the dog in the door frame until the
run timed out); people checked by time, not distance, along an edge; anybody
inside `tracking.NEAR_R` walking into the dog's way asked about; the dog's
front checked against walls, not only its centre; thresholds fixed for a
whole going-round, not re-taken at each replan (they ratcheted the dog onto
the cafeteria-door wall); and a waiting dog stepping sideways out of a
passer-by's line.

- [x] P2–P6 pass on both arms (2026-10-14)
- [x] Made `run_building`'s default, with the aware crowd (2026-10-15).
      `gate_d.py` still defaults to `--planner map`, for the reason above
- [x] Re-run at 10 Hz: `REPLAN_S` 0.2 → 0.1 s on 2026-10-15 (spec L6 asks
      ≥ 10 Hz). P2–P6 pass on both arms (`output/gate_d/20261015-115039`).
      Pedestrian contact on the VAMOS arm fell from 0.8 s at 5 Hz to 0; the
      dog goes round people a little more often (map arm 77 → 98)
- [ ] Margin re-measured on real people (`scripts/ce_rrt_calibrate.py` wants
      ground truth: a ceiling camera or a motion-capture room)

---

## Reporting KPIs (not pass/fail)

**Outcome:** arrival rate, time to goal, collisions, contacts, stops for
people and seconds spent waiting, seconds crawling, lift waits, spoken
refusals (stairs, unreachable floor).

**Dream behaviour:** VAMOS calls, candidates per call, candidates rejected by
the gate, fallbacks to the map route, mean factor of the chosen candidate,
commanded-speed throttle, dream latency, VLM latency.

If a value cannot be measured, write **N/A**, never 0.

---

## Fair comparison protocol

- [ ] Pass `--crowd` and `--planner` explicitly, never by default: `run_building`'s defaults moved to `aware` / `cerrt` on 2026-10-15, and every campaign before 2026-10-12 is `oblivious` / `map`. `summary.json` keeps both
- [ ] Freeze route, pedestrian seed, scene build (`output/scene_cache/building.xml` hash), code commit, `Dream` seed, `--speed`
- [x] **Pin the VLM.** The client asks for its candidates by beam search (`num_beams=5`, `temperature=0`), so the same frame gives the same 5 paths. Run both arms with a fixed `--vlm-latency` (e.g. 1.8): with it, a `--vamos` run repeats tick for tick; without it, answers land after the measured call time and runs drift. Do not pass `--vamos-sample` in a campaign
- [ ] Several seeds per route; report mean + dispersion or 95% CI
- [ ] A run that times out or is stopped by hand is **incomplete** — neither an arrival nor a clean run
- [ ] The trained parser in `models/lora/` is read-only during a campaign; experiments write under `output/`
- [ ] Record the SHA-256 of the scene and any model file before and after the campaign
