# Sim

MuJoCo digital twin of the building, with the planner and VAMOS in the loop.
Two scenes come out of the same occupancy grids the planner uses:

| scene | built by | driven by | covers |
|---|---|---|---|
| `output/scene_cache/floor1.xml` | `build_scene.py 1` | `run_demo.py`, `record_demo.py` | one floor |
| `output/scene_cache/building.xml` | `build_scene.py --building` | `run_building.py` | all three floors + the lift between them |

## The modules

Shared pieces, read by the entry points and by each other:

| file | what it owns |
|---|---|
| `control.py` | the control law and its gains -- also what `dreaming.py` imagines, so the dream and the dog cannot disagree |
| `clearance.py` | static clearance from the map: grid walls + Behavior Layer no-go zones |
| `overlay.py` | drawing onto the camera frame (goal pixel, candidate paths, LiDAR returns) |
| `vamos_client.py` | the VLM as an HTTP service: 5 candidate paths in, 1 chosen path out |

Subpackages:

| dir | what is in it |
|---|---|
| `scene/` | building the MuJoCo XML: `levels`, `build_scene`, `lift`, `stairs`, `obstacles` |
| `robot/` | the Go2 itself: `mujoco_robot`, `gait`, `robot_interface` |
| `sensing/` | what the dog perceives: `lidar` -> `perception` -> `dreaming` |

Entry points: `run_building.py` (the full stack), `run_demo.py` (one floor,
headless), `record_demo.py` (one floor, to MP4).

The `sensing/` order is the design: sensor, then interpretation, then
prediction. `lidar.py` reports surfaces and knows nothing about what they mean;
deciding which returns the map already explains is `perception.py`'s job. They
are kept apart so that the day a real Mid-360 arrives, only `lidar.py` changes.

## Whole building

```bash
python -m cyberdog.sim.scene.build_scene --building            # once; 552 geoms, floors 1-3
python -m cyberdog.sim.run_building "go to the server room"
python -m cyberdog.sim.run_building "library on the 2nd floor, then the server room"
python -m cyberdog.sim.run_building "restroom on floor 3" --no-video    # fast, headless
python -m cyberdog.sim.run_building "server room" --nlu                 # through the Gemma layer
python -m cyberdog.sim.run_building "server room" --vamos               # VLM steering
python -m cyberdog.sim.run_building "server room" --auto-confirm       # don't wait at the lift
python -m cyberdog.sim.run_building "server room" --speed 2            # play back 2x faster
```

### The one that shows everything

```bash
python -m cyberdog.sim.run_building \
  "take me to the library on the 2nd floor, then the stairs, then the server room" \
  --vamos --auto-confirm --speed 2
```

Three floors, two lift rides, a refusal and the VLM in the loop, in one
continuous run -- `output/scene_cache/building.mp4`, about 90 seconds at 2x:

| video | what it shows |
|---|---|
| 0:00 | multi-stop command split into three stops, floor phrase stripped, A* to the lift |
| 0:03 | lift 1 -> 2: stops, asks for the call button, rides, announces the floor |
| 0:11 | arrives at the library, announces, stands still |
| 0:11-0:42 | corridor at VAMOS's steering, candidate paths and safety bar live |
| 0:42 | **the stairs**: walks to the corridor outside them, turns to face them, and says it goes no closer |
| 1:12 | lift 2 -> 3 |
| 1:20 | server room |

`--speed` drops frames for playback only; control still runs at 20 Hz, and
the two 43 m corridor transits are what it is there for. The stairwell is at
the west end and the lift at the east, so visiting both is a round trip.

Not in this video: the Gemma layer (`--nlu`). torch is deliberately not in the
sim environment -- the VLM is a service, not an import -- so the NLU layer is
demonstrated by `main_planner.py` in its own process. The stop splitting and
floor parsing in the video are the same rules that path uses.

`--vamos` needs `VAMOS/server/vlm_server.py` running on 127.0.0.1:8009.

The command is split into stops and stripped of floor phrases by the same
rules `main_planner.py` uses. `--nlu` sends each stop through the fine-tuned
Gemma layer; without it the destination is matched against `locations.json`
by name, which keeps torch out of the process.

`BuildingRouter` returns one leg per floor with a "take the lift" handover --
`run_building.py` is what actually drives it: A* leg, ride, A* leg, ride, A*
leg, in one continuous run. Video is `output/scene_cache/building.mp4`: chase view on
the left with a three-pip floor indicator, the dog's own 640x480 camera on the
right with the projected goal pixel (and VAMOS's candidate paths under
`--vamos`).

## The lift, and the stairs it refuses

This robot guides someone who cannot see the steps and has one hand on the
handle. Stairs are a hazard, not a route. So:

- `lift.py` holds the shaft at the east dead-end of the corridor, and both the
  scene builder and the driver read it -- the car cannot stop at a height the
  scene has no opening at. Its footprint is cut out of every upper slab, and
  the boarding point is `elevator` in `locations.json`, read at import so the
  car and the map agree.
- `stairs.py` still draws the flights at the *west* end, and nothing climbs
  them. They are there so the dog's camera and VAMOS's candidate paths have
  the real hazard to be scored against. The two ends are opposite so a route
  heading the wrong way is wrong on sight.
- The stairwells are `stairs` zones in `data/building/zones.json`,
  which `BuildingRouter` stamps into its planning grids as obstacles. No A*
  route can contain one, `free_space_test()` reports them as not free so the
  VAMOS confidence gate rejects candidates pointing into one, and `follow()`
  halts outright (`HazardStop`) if the dog is ever inside one.

Boarding is a handover, not an autonomy claim: the dog stops short, says
"press the call button for floor N, then press continue", and waits. The real
robot cannot press a button, so a human does. `--auto-confirm` answers for
them after a visible pause, which is what batch runs and recordings use.

The ride itself is scripted: `cmd_vel` has no z, so the dog is placed along
the car's travel and the velocity loop picks up again on the next floor. The
few metres in and out of the car are scripted for the same reason -- the
velocity controller would otherwise turn on the spot inside a shaft it cannot
see.

## The safety factor

VAMOS proposes, and before the dog commits to a proposal it imagines driving
it. `dreaming.py` rolls the real control law forward from the current pose
along each candidate path, 24 times, with noise on the things that are
actually uncertain -- heading bias, speed, per-step slip. The fraction of
imagined runs that get to the end without hitting anything is a collision
probability; weighted by how much room they left, that is the safety factor:

    safety = P(no collision) x (0.4 + 0.6 x room)

where `room` is the mean closest approach scaled to [0, 1] between the robot
radius (0.16) and 0.40 m, so a run that survives but scrapes still keeps 0.4
of its score. It is in the spirit of the spec's chance constraint (CE-RRT*,
P_coll < 0.01) reached by sampling rather than algebra -- but not that
constraint: the gate at 0.5 lets through a path with P_coll = 0.2, and 24
rollouts cannot resolve 0.01 anyway. It does three things: rejects anything under
0.5 (the confidence gate), ranks what survives safest-first and
progress-second, and scales the dog's speed, so a path it only just believes
in is walked at half pace. The map's own A* route is the fallback and is not
throttled -- it came from the ground-truth grid, not from a picture.

Scores from the real thing, floor 1:

| candidate | safety | why |
|---|---|---|
| straight down the corridor | 1.00 | never came within a metre of anything |
| through a doorway | 0.59 | survives 92% of the time, but with 0.26 m to spare |
| hugging a wall | 0.28 | 46% of imagined runs clipped it — rejected |
| aimed into a wall | 0.00 | every imagined run hit something |
| aimed into the stairwell | 0.00 | same — the factor refuses stairs on its own |

That last row is the one worth having. The stairwell is not in the occupancy
grid; it is in the Behavior Layer, and `clearance_test()` folds both into one
distance field, so the imagined dog falls down the imagined stairs and the
real one never goes near them.

Both halves of the number are visible while it runs: the bar at the bottom of
the dog's camera panel, and a per-floor summary at the end of the run
("23 candidates rejected, 8 of them by the safety gate, mean safety of the
paths it followed 0.67").

Two honest limits. The rollouts use the same kinematic model the twin is built
on, so they inherit its optimism -- no gait, no contacts, no real dynamics.
And the clearance field is built from grids that were already inflated by the
robot's radius, so its distances run about 0.25 m short of the truth. Both err
towards caution, which is the direction to err in.

## Walking

The dog is still driven kinematically -- body velocities in, pose out, no
contacts -- but the legs are no longer frozen in the home stance. `gait.py`
poses them from the distance the body has actually covered: a trot, diagonal
pairs together, each foot tracing a flattened oval found by two-link inverse
kinematics on the 0.213 m thigh and calf. Turning on the spot feeds the same
phase, so a pivot steps instead of pirouetting on planted feet, and the
amplitude fades in and out so a stopped dog settles into the home stance.

Nothing here holds the dog up and nothing here is a controller -- on the real
Go2 the gait belongs entirely to the onboard locomotion policy. It is an
honest drawing of walking, which is what the video needed: before this, anyone
watching had to take it on trust that this was a robot that walks.

## What the map does not know

Every other solid in the scene is also in the occupancy grid the planner runs
A* on, which makes the perception half of the stack decorative: VAMOS proposes
paths from a picture and they are scored against the same ground truth the
route already came from, so the camera cannot contribute anything the map did
not have.

`obstacles.py` is the exception. Four boxes -- one on floor 1, two on floor 2,
one on floor 3 -- are drawn into the scene and deliberately left out of
`data/building`: no grid cell, no zone, no location. The map is stale
on purpose. Checked against the router, all four stand in space the grid calls
free, and all four have real A* legs running straight through them:

| floor | box | A* leg that hits it |
|---|---|---|
| 1 | trolley, x 23.4-24.6 | `room 108`, `hallway` |
| 2 | crate, x 34.0-35.2 | `room 201`-`204`, `electrical engineering lab` |
| 2 | cart, x 16.0-17.2 | `room 201`, `room 206`-`208` |
| 3 | cartons, x 41.0-42.2 | `biology lab`, `physics lab`, `restroom`, `room 301` |

Where in the corridor matters more than it looks, and there are two lines to
miss. A*'s checkpoints hug whichever wall the destination is on (y=8.6
southbound, y=10.3 northbound), but the dog does not drive the polyline --
pure-pursuit cuts the corners and the long transits are walked down the middle
at y=9.5. A box that clears both is in the corridor and never in the dog's way:
a reassuring video and no experiment. So each box spans from its own side
across the centreline, which `obstacles.py` asserts rather than hopes for, and
its position along the corridor follows the routes that actually cross that
stretch. Re-check both after any change to the grids or the router.

They are still absent from the map. What changed is that the dog can now see
them -- `lidar.py` and `perception.py` below -- and goes round.

## Seeing them, and going round

`lidar.py` is the virtual Mid-360 the spec asks for at L2 step 2: ~2000 rays,
360 degrees, **1.7 ms a scan**, emitting the `/lidar/points`-shaped cloud L5 and
L6 are specified to consume, so the real unit swaps in behind one interface.

`perception.py` turns a scan into the one thing the rest of the stack already
understands. It keeps returns between ankle and head height, throws away
everything the static field already explains -- walls, stairwells, the lift
shaft -- and stamps what is left into a small local distance transform fused
with the static one. Measured over the whole floor-2 corridor: **zero false
positives**, every unexplained return coming from a known box.

Two details that are not fine-tuning:

- A range sensor sees *surfaces*. The crate comes back as the one face pointing
  at the dog, and the space behind it is unknown, not free. Each return shadows
  a metre of the ray behind it, or the inside of a box reads as open floor and
  goals get placed in the middle of one.
- Detected things and mapped things are asked about separately. Every doorway
  in the building is tight in the static field and A* routed through it
  anyway; treating that as an obstruction means never leaving a room.

### Who actually does the avoiding, and why

VAMOS drives at whatever goal pixel it is handed, and the carrot comes from A*
on a map with no crate in it -- so the pixel lands *on* the crate. Measured
before any of this was built: at 0.8 m the crate fills 60% of the frame and all
five candidates still go straight through it. That is not a model that cannot
avoid obstacles, it is a model being told to walk into one, and rejecting all
five leaves nothing to follow.

So the map's half of "map decides WHERE, VLM decides HOW" uses the sensor:
`free_carrot` moves the goal sideways into the gap. Which side is decided at
the *pinch* -- the tightest point on the way -- because judged at the carrot
nothing looks blocked until the dog is level with the crate, and judged beyond
it the roomiest direction is "straight on". How far is decided at the carrot,
because the shift that centres the pinch puts a carrot that is already past the
crate into the corner it just cleared.

While a detour is on, that displaced goal steers and VAMOS keeps proposing and
being judged -- its candidates and the gate's rejections are both in the video.
That split is a measurement, not a demotion: zero-shot VAMOS spreads its five
candidates about **+/-0.25 m over a 2 m path**, and getting round a crate in
this corridor takes about **0.65 m**. Its paths are safe over their own length,
so they pass the gate and still walk the dog into the thing -- the gate cannot
reject a path that is fine for 2 m and leads nowhere at 4 m. Spec L4 step 5,
the LoRA fine-tune, is what would let the model make this turn itself.

### When there is no way past

Three steps, not one rule. Crawl at 30% while re-asking VAMOS sooner than the
usual 1 Hz; stop when the dog is touching distance from something with nothing
approved; and after 8 s of that, say so and end the leg rather than grinding on
in silence. Announcements come *before* the lean, not during -- the person's
arm is on the handle and which way it is about to go is the one thing they
cannot see coming.

### What it costs

| | |
|---|---|
| LiDAR scan + fused costmap + displaced goal | **3.2 ms** per control tick (6% of a 20 Hz tick) |
| `dreaming.py`, 5 candidates x 24 rollouts | **7.4 ms** per replan |
| the VLM call in the same loop | **~1.8 s** |

### Scored on ground truth

`run_building.py` compares the dog's pose each tick against `obstacles.boxes()`
-- ground truth, never shown to the robot, which finds these with the LiDAR or
not at all -- and prints the result. Every run below ends `collisions: none`:

```
python -m cyberdog.sim.run_building "room 201" --vamos --auto-confirm --speed 2
  ARRIVED on floor 2 after 102s
  VAMOS floor 2: 87 calls, 0 failed, 177 candidates rejected (40 by the safety gate)
  collisions: none -- the dog never entered an obstacle's footprint
```

`room 101` (floor 1), `room 201` (floor 2), `chemistry lab` (floor 3) and
`cafeteria` all arrive with zero collisions and zero stops.

## Known limits

- The dog is kinematic -- no contacts, and the gait is drawn, not simulated.
- The obstacles are seen, but there is no memory: each scan stands alone, which
  is fine for a 360-degree sensor and wrong the moment something is occluded.
  Beyond LiDAR range an imagined rollout is still scored on the static map.
- People and anything that moves are not in the scene at all (spec L2 step 5),
  and CE-RRT* (L6 §2) does not exist -- VAMOS proposes and the gate disposes.
- The lift has no doors and no call delay: the car is always where the dog is.
- The Gemma layer was fine-tuned before the lift existed, so `--nlu` cannot
  parse "take me to the lift" as a destination -- it maps unknown words onto
  the nearest name it knows. That is why hazard names are checked against the
  map with rules *before* the model runs (`BuildingRouter.hazard_named`), and
  why "take me to the stairs" (or staircase / stairway / stairwell) walks to
  the corridor outside them and says it goes no closer, on either path. Add
  `elevator`/`lift` to `cyberdog/language/generate_dataset.py` on the next
  regeneration.
