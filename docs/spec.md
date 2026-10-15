# Cyberdog Blind Navigation — PoC Development Skill
## Map-Grounded VAMOS + RDog + CE-RRT* | Zero-Hardware Proof of Concept

---

## 0. How to Use This File

- **Audience:** coding agent / developer starting the PoC.
- **Golden rule:** build every layer against **mocks first**, then swap in real modules one at a time. Never debug two untested layers at once.
- **Scope (PoC):** a closed-loop *"go to the library"* demo running in a **digital twin** (synthetic camera/LiDAR/pose), with the real VAMOS model in the loop. **No robot hardware required.**
- **Out of scope (later phases):** real Cyberdog locomotion wiring, real LiDAR rig, physical Smart Handle, blind-user trials.
- **Design principle:** *AI proposes, simple code disposes.* All safety-critical decisions are deterministic rules; all open-world understanding lives in AI modules.

---

## 1. Context

**Goal:** navigation system on a Xiaomi Cyberdog (Unitree Go2 kinematics) that guides blind students across campus — a robotic guide dog.

**Core papers:**
| Paper | Role in our architecture |
|---|---|
| **VAMOS** (arXiv:2510.20818) | The "brain": hierarchical VLA. VLM planner (PaliGemma 2 3B) outputs 5 candidate 2D paths; affordance MLP re-ranks by traversability. 90% real-robot success. |
| **RDog** (CHI 2024, 10.1145/3613904.3642227) | The "body interface + map": FAST-LIO 3D map + Behavior Layer (semantic zones), rigid Smart Handle (load cell + joystick), preemptive voice feedback. |
| **CrossTracer / NaviTrace** (arXiv:2608.06688 / 2510.26909) | The benchmark to beat (CrossTracer score 45.68). NaviTrace val split = fine-tuning signal + diagnostic. |

**Finalized architectural decisions:**
0. **The lift first; the stairs only as a fallback the person agrees to.** The user cannot see the steps and has one hand on the handle, so a staircase is never something a route wanders onto: stairwells are full-stop zones in the Behavior Layer, stamped into the planning grids so **no A\* route can contain one**, and any unplanned drift into one is a hard stop with no resume. Vertical movement is by **lift** by default, and boarding it is gated on human confirmation — the robot cannot press a call button and does not pretend to.
   *Revised (Oct 2026):* this rule used to read "stairs are never traversed". Blind people do use stairs, and a guide dog's job is to make them safe, not to forbid them. So a floor change may now use the stairs as a **deliberate transfer, like the lift ride**, and only when (a) the lift is out of service or does not serve the floor, or (b) the lift route is both **2× and 30 m longer** than the stairs route (`STAIRS_RATIO`, `STAIRS_EXTRA_M`). It is never taken silently: the dog **asks twice** — when it is about to set off for the stairs (why not the lift, how many steps, up or down, which hand the rail is on; no means the lift), and again **stopped at the first step** — then goes at a third of corridor pace, announcing every landing and the last step. `--no-stairs` restores the old rule. A floor neither can reach is unreachable, and the honest answer is to say so.
   *Refusing to move is not the answer either.* Asked for a place inside a no-go zone, the robot walks to the nearest safe point outside it — the corridor by the stairs — and the arrival announcement says plainly that this is as close as it goes. Answer the question that was asked; decline only the part that is unsafe.
1. **Map decides WHERE** (3D map + Behavior Layer + A*), **VLM decides HOW** (VAMOS local paths), **rules decide WHEN TO STOP**.
2. **VAMOS requires a goal pixel pointer** → solved by the **Checkpoint Projector** (pinhole projection of an intermediate destination 2–4 m ahead on the map route). Pure geometry, zero AI.
3. **One PaliGemma 2 3B backbone, multiple LoRA adapters:** `nav` (VAMOS paths), `goal` (instruction → endpoint pixel), `intent` (free-form voice → slots), `describe` (optional terrain captions).
4. **Affordance MLP** trained in Isaac Lab (elevation + position + heading → traversability prob); **confidence gate rejects < 0.5**. The gate is fed by a **safety factor**, not a raw traversability score: the robot imagines driving each candidate with its own controller and noise, and the factor is P(no collision) × clearance margin. Judge the plan the robot would execute, not the line the VLM drew.
5. **CE-RRT\*** (chance-constrained, P_coll < 0.01, warm-started) for dynamic obstacle avoidance.
6. **Task Planner = deterministic finite state machine.** Not an LLM. It calls AI modules as subroutines.
7. **ROS split:** VAMOS runs ROS 1 (Noetic) in a container; everything new runs ROS 2 (Humble); `ros1_bridge` connects them.

**VAMOS repo facts (`github.com/vamos-vla/vamos-deployment`):** inference/deployment ONLY (ROS 1 workspace, demo scripts, VLM server). **No training code.** We reuse its inference server + message definitions; we re-implement all training and all other layers.

---

## 2. Architecture Overview

```
User Interface (voice, Smart Handle mock, haptic mock)
        ↓
Ground Truth Map (3D mesh/.pcd → occupancy grid → Behavior Layer zones → A* route + checkpoints)
        ↓
Sensor Simulator (digital twin: virtual RGB @0.5m, virtual LiDAR, GT pose)   [real sensors later]
        ↓
Task Planner (FSM: intent parser → A* checkpoints → CHECKPOINT PROJECTOR → goal pixel pointer)
        ↓
Enhanced VAMOS Core (VLM planner: image+pixel+voice → 5 paths → Affordance MLP re-rank)
        ↓
Shared Control & Safety (handle override · confidence gate · CE-RRT*)
        ↓
Low-Level Execution (PoC: kinematic executor | later: Cyberdog native locomotion RHC 2Hz)
        ↓  + Preemptive Voice Engine (Behavior Layer triggers)
Cyberdog Executes → haptic/audio feedback loop
```

---

## 3. Layer-by-Layer Development (Step by Step)

### LAYER 1 — Ground Truth Map + Behavior Layer (no hardware)
**Purpose:** the deterministic "WHERE" source: localization prior, global routes, semantic zone triggers.

**Data sources (pick one):**
- Phone photogrammetry: walk campus filming → **Scaniverse / Polycam / Luma AI / RealityScan** → export mesh (.obj/.ply) or point cloud. *(Best: real campus.)*
- **Google Photorealistic 3D Tiles** (if city covered) → convert to mesh.
- Public stand-ins for pipeline testing: **NCLT** (campus LiDAR), **HM3D / Stanford 2D-3D-S** (indoor).

**Steps:**
1. Clean mesh (decimate, fill ground holes) in CloudCompare/MeshLab.
2. Convert to `.pcd`; project ground plane → rasterize → **2D occupancy grid** (`nav_msgs/OccupancyGrid`).
3. Annotate **Behavior Layer**: polygons with tags `{grass: slow+announce, stairs: STOP+announce, crosswalk, entrance_zone}` → save as JSON. Zones are generated from the same constants the geometry is built from, so a zone cannot end up somewhere the hazard isn't.
4. Implement **A\*** over occupancy grid; simplify polyline with Ramer–Douglas–Peucker → **checkpoints** (turn points double as voice-announcement triggers). **Full-stop zones are stamped into the planning grid as obstacles** before A* runs — a no-go area is not something to route through and halt at, it is something no route may contain.
5. Publish the **lift** (or lifts) as named transit nodes, and the **stairs** as the fallback one. Every cross-floor route is `current floor → lift → target floor`, or `→ stairs →` when rule 0's fallback applies; walking routes never cross a stairwell either way.

**PoC stub:** none needed — this layer is fully doable now.
**Acceptance:** occupancy grid + A* route + checkpoints render correctly in RViz for ≥3 route queries.
**Feeds:** Task Planner (L3).

### LAYER 2 — Sensor Simulator / Digital Twin + Auto-Labeler
**Purpose:** replace all hardware with synthetic streams; generate free training labels.

**Steps:**
1. Import mesh into **Isaac Sim** (preferred; already needed for affordance) or Habitat-sim/Blender.
2. Add virtual sensors: RGB camera at **0.5 m** with known intrinsics `K`; LiDAR raycaster (Mid-360 pattern); ground-truth pose publisher.
3. Drive a virtual agent along A* polylines; publish `/camera/image_raw`, `/lidar/points`, `/odom` as rosbags/live topics.
4. **Auto-Labeler (killer feature):** using GT pose + `K`, project route waypoints into each rendered frame → dataset of `(image, goal_pixel, path_pixels, preference_text)` — LoRA training data with zero teleoperation.
5. Add 2–3 synthetic walking pedestrians for CE-RRT* testing.

**PoC stub:** kinematic agent motion is fine (no physics needed here).
**Acceptance:** ≥200 rendered dog-height frames; projected pixel overlay lands on intended target in ≥9/10 checked frames.
**Feeds:** VAMOS (L4), Projector (L3), CE-RRT* (L6), training pipeline.

### LAYER 3 — Task Planner + Checkpoint Projector (deterministic FSM)
**Purpose:** orchestration + the goal-pixel pointer VAMOS requires. **~90% of this is simple code.**

**States:** `IDLE → GLOBAL_NAV → ALIGN → LOCAL_NAV → LIFT_WAIT → LIFT_RIDE → ARRIVED` (or `→ STAIRS_OFFER → STAIRS_WAIT → STAIRS_WALK →`) + interrupts `EMERGENCY_STOP`, `STAIRS_STOP`, `HOLD`.
`LIFT_WAIT` is a stop with a spoken handover ("press the call button for floor N, then press continue") and does not advance without human confirmation. `STAIRS_OFFER` and `STAIRS_WAIT` are the two confirmations of rule 0's fallback. `LIFT_RIDE` and `STAIRS_WALK` are the only states in which the robot's height changes. `STAIRS_STOP` — walking into a stairwell without having agreed to — has no resume: it is a halt, not a confirm-and-continue.
**Sub-modules:**
1. **Intent parser:** lookup/fuzzy match of destination against Behavior Layer node list. (PaliGemma `intent` adapter optional later.)
2. **Route consumer:** reads A* polyline + checkpoints from L1.
3. **Checkpoint Projector:** sample destination 2–4 m ahead; project:
   `P_cam = T_body→cam⁻¹ · T_map→body⁻¹ · P_map` ; `(u,v) = project(K, P_cam)`;
   valid if depth > 0 and inside FOV margin; else enter `ALIGN` (rotate until visible).
   **Fallback:** PaliGemma `goal` adapter for unmapped targets ("find a bench").
4. **Preference generator:** emits language steering from next checkpoint (e.g. *"turn right after the door, stay on dry pavement"* — include standing preferences: avoid puddles, stay on sidewalk).

**Interfaces out:** `/vamos/goal_pixel` @1–2 Hz, `/voice/feedback` triggers, `/route/polyline`.
**PoC stub:** voice input = text file/keyboard; handle = keyboard keys.
**Acceptance:** with replayed L2 streams, projector emits a smoothly moving in-frame pixel for a full route; FSM logs show correct transitions incl. ALIGN at a blind corner.
**Feeds:** VAMOS (L4).

### LAYER 4 — Enhanced VAMOS Core (repo reuse + LoRA)
**Purpose:** semantic local planner ("HOW").

**Steps:**
1. Clone `vamos-deployment`; audit `vamos_ws/src/` message schemas; run `server/` + `vamos_demo.py` on a lab RTX 4090 (ROS 1 Noetic container).
2. Set up `ros1_bridge`: `/camera/image_raw` ROS2→ROS1; `/vamos/candidate_paths` ROS1→ROS2.
3. Wire projector output into the VLM prompt: **(image + goal pixel + preference)** → receive **5 candidate 2D paths**.
4. **Zero-shot test first** on L2 rendered frames (pass criterion: ≥1 path stays on pavement toward pointer).
5. **LoRA fine-tune** (reimplement; repo has no training code): config per paper — targets `q,k,v,o,gate,up,down_proj`, **r=16, α=16, dropout=0.05**, start from HF checkpoint; data = L2 auto-labeled pairs + NaviTrace val (legged-robot subset) + rainy/puddle scenes.
6. Add `goal` adapter (instruction → endpoint pixel) as second LoRA on same backbone.

**PoC stub:** zero-shot checkpoint is acceptable for first closed-loop test; fine-tune is the upgrade.

**Decision (2026-10-14): steps 5–6 are deferred to the hardware phase; the PoC ships with base VAMOS.** Fine-tuning on the twin's rendered frames teaches the model the twin, not the building: the frames are synthetic and the labels are the twin's own walked paths. The one adapter trained (`scripts/vamos/README.md`) did not lower goal-pixel error with the live prompt (0.231 → 0.230), drew near-straight lines to the prompt's goal, and failed Gate D on D2 and D3 (`docs/dreaming_safety_kpis.md`, 2026-10-11). The training and scoring code stays in `scripts/vamos/`; it is re-run on real camera frames from the brain rig, which is the data a fine-tune is worth doing on.

**Acceptance:** end-to-end image+pixel → 5 paths at ≥0.5 Hz; paths track the moving destination.
**Feeds:** Affordance (L5).

### LAYER 5 — Affordance Module (Isaac Lab)
**Purpose:** physics filter — knows stairs/wet/rough are not walkable even if the VLM likes them.

**Steps:**
1. Isaac Lab + Unitree Go2 URDF; procedural terrains (stairs, ramps, mounds, wet-flag patches).
2. Roll out **native locomotion policy**; log (local elevation map, position, heading, success/fail).
3. Train **MLP** from scratch, loss = binary cross-entropy, output traversability ∈ [0,1].
4. Runtime node: build local elevation from `/lidar/points`; score the 5 VAMOS candidates; **re-rank**; publish `/vamos/safe_path`; confidence gate rejects < 0.5.

**PoC stub:** a clearance field (walls from the occupancy grid + Behavior Layer stop zones, as one distance transform) to unblock integration; swap trained MLP after. It answers the same question the MLP will — "can the dog be here?" — from the map instead of from elevation, and gives the MLP something to be measured against.
**Acceptance:** held-out sim terrain accuracy > 90%; a stair-crossing candidate is never the top-ranked path.
**Feeds:** Safety (L6).

### LAYER 6 — Shared Control & Safety
**Purpose:** the differentiator. Human override + probabilistic dynamic avoidance.

**Sub-modules:**
1. **Handle mock → later ESP32:** `/handle/force` @50 Hz; rules: slight pull = decel, slight push = accel, **tug = EMERGENCY STOP** (hard override of everything).
2. **CE-RRT\* planner:** reference = `/vamos/safe_path`; obstacles = live `/lidar/points` (+ predicted pedestrian motion); chance constraint P_coll < 0.01; **warm-start** from previous tree; outputs `/cmd_vel_safe` @10–20 Hz.
3. **Imagined rollouts ("dreaming") → the safety factor.** Before committing to a VAMOS candidate, roll the robot's *own control law* forward along it from the current pose, N times, with noise on heading bias, speed and per-step slip. `safety = P(no collision) × (clearance margin)`. This is the same chance constraint as CE-RRT*, reached by sampling instead of algebra, and it makes the constraint about *what the robot would actually do* rather than about the line the VLM drew. The factor **gates** (< 0.5 rejected, the confidence gate), **ranks** (safest first, progress second) and **throttles** (commanded speed scales with it). Deterministic map routes are the fallback and are not throttled. Implemented in `sim/sensing/dreaming.py`; the upgrade path replaces the kinematic rollout with the real locomotion model and the clearance field with the affordance MLP, leaving the rest as is.
4. **Safety mux:** merges CE-RRT* velocity with handle override + confidence gate + LiDAR hard-stop bubble → `/cmd_vel_final`.

**PoC stub:** plain RRT* or DWA first; upgrade to chance-constrained version second.

**Status (2026-10-15): CE-RRT\* built, and the default since 2026-10-15** (`--planner cerrt`; `--planner map` waits instead) (`src/cyberdog/planning/ce_rrt.py`). RRT\* in space and time, curves not pivots, warm-started, replanned every 0.1 s (10 Hz, raised from 5 Hz on 2026-10-15), about 15–40 ms a plan. It goes round people only; crates stay with `free_destination`. The chance constraint is calibrated, not assumed: the tracker's straight-line prediction is out by 0.64 m (99th percentile) half a second ahead and 1.3 m a second ahead (`scripts/ce_rrt_calibrate.py`), so P(contact) < 0.01 is held over the committed 0.5 s, with the fallback -- waiting -- for when no plan meets it. A waiting dog steps sideways out of a passer-by's predicted line. Judged by its own paired gate (`scripts/planner_ab.py`, P2–P6 on Gate D's bars): **passes on both arms**, waiting for people −53% map-only and −42% `--vamos`, contact 1.7 → 0.0 s and 1.2 → 0.8 s (`docs/dreaming_safety_kpis.md`). It does not avoid every pedestrian without a stop: 80–106 stops remain per 70 runs, most of them (66 of 86 in one map-only sweep) where no path keeps the 1% margin in a 2.5 m corridor. Better prediction, not more tuning, is what removes those.

**Acceptance:** avoids a synthetic pedestrian without full stops; tug → zero velocity in < 100 ms; stair zone → STOP + voice **and no planned route ever enters one**; a cross-floor route is refused out loud when no lift serves the target floor; a candidate path aimed at a stairwell scores 0.00 and is rejected by the gate, without anything stair-specific being coded into the scorer.
**Feeds:** Execution (L7).

### LAYER 7 — Execution + Feedback (PoC: kinematic)
**Purpose:** close the loop; announce terrain.

**Steps:**
1. **Kinematic executor (PoC):** integrate `/cmd_vel_final` to move the virtual agent in the twin (this updates `/odom` and closes the loop). *Later phase replaces this with Cyberdog native locomotion (RHC 2 Hz) via ROS 2 driver.*
2. **Preemptive Voice Engine:** pose-in-zone checks against Behavior Layer → TTS strings ("grass ahead", "turning right", "stairs detected — stopping. I only take the stairs when you have agreed to", "the lift is out of service. I can take you by the stairs: 14 steps up, handrail on your right", "lift ahead — press the call button for floor 3, then press continue"). Checkpoint lines are spoken *before* the turn they describe; arrival is spoken on arrival.
3. **Haptic mock:** vibration command on turn/stop (keyboard LED / log now; vest/handle motor later).

**Acceptance:** full loop stable for a complete route; voice triggers fire at correct zones.

---

## 4. Connection Plan (ROS Graph & Interfaces)

| Topic | Type | Rate | Producer → Consumer |
|---|---|---|---|
| `/voice/command` | String | event | UI mock → task_planner |
| `/handle/force`, `/handle/joystick` | Float32 / Joy | 50 Hz | handle_mock → safety_mux / task_planner |
| `/map/occupancy`, `/map/zones` | OccupancyGrid / JSON | static | map_tools → task_planner, voice_engine |
| `/odom` | Odometry | 50 Hz | sim (later FAST-LIO) → projector, safety, planner |
| `/camera/image_raw` + `/camera/info` | Image / CameraInfo | 10–30 Hz | sim → VAMOS (bridge), projector debug |
| `/lidar/points` | PointCloud2 | 10 Hz | sim → elevation builder, ce_rrt |
| `/route/polyline` | Path | 1 Hz | task_planner → projector |
| `/vamos/goal_pixel` | PointStamped | 1–2 Hz | projector → VAMOS server (bridge) |
| `/vamos/candidate_paths` | PathList | 0.5–1 Hz | VAMOS (ROS1) → affordance (bridge) |
| `/vamos/safe_path` | Path | 0.5–1 Hz | affordance → ce_rrt |
| `/cmd_vel_safe` → `/cmd_vel_final` | Twist | 10–20 Hz | ce_rrt → safety_mux → executor |
| `/voice/feedback`, `/haptic/cmd` | String / UInt8 | event | voice_engine/safety → UI |

**Bridge:** `ros1_bridge` limited to the two VAMOS crossings above (keep bridge surface minimal).
**Integration order:** L1 → L2 → L3 → L4 → L5 → L6 → L7, testing each new node against the previous layer's *recorded* topics before going live.

---

## 5. Six-Week PoC Schedule & Exit Gates

| Week | Work | Exit gate | Status (2026-10-15) |
|---|---|---|---|
| 1 | L1 map pipeline + L2 twin setup | Route + checkpoints in RViz; 200 rendered frames | done (MuJoCo twin, not Isaac Sim; routes checked as PNGs, not RViz) |
| 2 | L3 planner/projector + L4 VAMOS zero-shot wiring | Moving pixel pointer; VAMOS returns 5 paths on twin | done |
| 3 | ~~L4 LoRA (auto-labeled data)~~ deferred to hardware (see L4) + L5 affordance stub→MLP | ~~Fine-tuned goal pixel error ↓~~; stair candidate never wins | stub done (a stair candidate scores 0.00, Gate B4); MLP trained in Isaac, running in shadow mode only |
| 4 | L6 safety + CE-RRT* (+pedestrians) | Smooth avoidance; tug < 100 ms | done: CE-RRT\* passes its gate and is the default; a tug stops the dog on the tick it is felt (50 ms). Not every pedestrian is avoided without a stop |
| 5 | L7 closed loop + voice; full-route runs | ≥80% route success over 10 runs; latency budget measured | success done (Gate D: 69–70 of 70 per arm); latency measured piecewise (`docs/architecture.md`), not yet written up against the budget |
| 6 | Eval report + optional NaviTrace diagnostic | **Go/No-Go** for hardware phase | not started |

**Latency budget to measure:** VAMOS ≤ 2–3 s/plan (known), projector 1–2 Hz, CE-RRT* ≥ 10 Hz, safety mux 50 Hz, handle stop < 100 ms.
Measured so far: VAMOS ~1.6–1.8 s per call on an M-series Mac; projector and VAMOS asked at 1 Hz; CE-RRT\* 15–40 ms a plan, replanned at 10 Hz (`REPLAN_S` = 0.1 s, raised from 5 Hz on 2026-10-15; the planner's gate is to be re-run at this rate); the safety mux runs at the 20 Hz control rate, not 50 Hz, so a tug stops the dog within one 50 ms tick.

---

## 6. Reuse vs Build

| From VAMOS repo (reuse) | We build (not in repo) |
|---|---|
| VLM inference server, ROS1 ws, demo scripts, msg defs | LoRA training (+ adapters), affordance training, task planner, projector, map pipeline, sensor twin, safety/CE-RRT*, handle/voice/haptic mocks, ros1_bridge config |

**Suggested repo layout:** `cyberdog-poc/{map_tools, sim_twin, task_planner, vamos_bridge, affordance, safety, execution, docs}`.

---

## 7. Definition of Done (PoC)

- [x] Closed-loop *"go to the library"* completes in the digital twin with zero hardware. (2026-10-15, PoC defaults, three people: `library` and `library on the 2nd floor` both arrive with no collision or contact.)
- [x] A multi-floor run completes with **zero unplanned stair traversal**: the lift is announced, confirmed and ridden; stairs are used only through rule 0's fallback, after both confirmations. (Gate D, 2026-10-14: 140 runs, D4 stair / no-go entries 0.)
- [ ] Every layer's failure is individually logged and interpretable (which layer rejected what).
- [ ] Latency + success metrics recorded; go/no-go report written.
- [ ] **Only then** buy hardware: Jetson Orin NX 16 GB, Livox Mid-360 (~$800), USB cam, ESP32 handle (~$40) — the brain rig phase.