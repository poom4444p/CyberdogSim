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

- [ ] D1–D8 all pass
- [ ] Promotion is an explicit decision; never promote from selftest alone

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

- [ ] Freeze route, pedestrian seed, scene build (`output/scene_cache/building.xml` hash), code commit, `Dream` seed, `--speed`
- [x] **Pin the VLM.** The client asks for its candidates by beam search (`num_beams=5`, `temperature=0`), so the same frame gives the same 5 paths. Run both arms with a fixed `--vlm-latency` (e.g. 1.8): with it, a `--vamos` run repeats tick for tick; without it, answers land after the measured call time and runs drift. Do not pass `--vamos-sample` in a campaign
- [ ] Several seeds per route; report mean + dispersion or 95% CI
- [ ] A run that times out or is stopped by hand is **incomplete** — neither an arrival nor a clean run
- [ ] The trained parser in `models/lora/` is read-only during a campaign; experiments write under `output/`
- [ ] Record the SHA-256 of the scene and any model file before and after the campaign
