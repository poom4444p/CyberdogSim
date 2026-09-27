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
| `scene/` | building the MuJoCo XML: `levels`, `build_scene`, `lift`, `stairs`, `obstacles`, `pedestrians` |
| `robot/` | the Go2 itself: `mujoco_robot`, `gait`, `robot_interface` |
| `sensing/` | what the dog perceives: `lidar` -> `perception`/`tracking` -> `dreaming` |

Entry points: `run_building.py` (the full stack), `run_demo.py` (one floor,
headless), `record_demo.py` (one floor, to MP4).

The `sensing/` order is the design: sensor, then interpretation, then
prediction. `lidar.py` reports surfaces and knows nothing about what they mean;
deciding which returns the map already explains is `perception.py`'s job. They
are kept apart so that the day a real Mid-360 arrives, only `lidar.py` changes.

## Whole building

```bash
python -m cyberdog.sim.scene.build_scene --building            # once; 390 geoms, floors 1-3
python -m cyberdog.sim.run_building "go to the server room"
python -m cyberdog.sim.run_building "library on the 2nd floor, then the server room"
python -m cyberdog.sim.run_building "restroom on floor 3" --no-video    # fast, headless
python -m cyberdog.sim.run_building "server room" --nlu                 # through the Gemma layer
python -m cyberdog.sim.run_building "server room" --vamos               # VLM steering
python -m cyberdog.sim.run_building "server room" --auto-confirm       # don't wait at the lift
python -m cyberdog.sim.run_building "server room" --speed 2            # play back 2x faster
python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --seed 1  # people in the corridors
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

Not in this video: the Gemma layer (`--nlu`), though `run_building.py` can drive
it. The VLM stays a service rather than an import, but the command parser is
small enough to run in-process, so it is an optional extra on this same
environment (`pip install -e ".[language]"`) rather than a server of its own.
The stop splitting and floor parsing in the video are rules either way.

`--vamos` needs `VAMOS/server/vlm_server.py` running on 127.0.0.1:8009.

The command is split into stops and stripped of floor phrases by the same
rules `main_planner.py` uses. `--nlu` sends each stop through the fine-tuned
Gemma layer; without it the destination is matched against `locations.json` by
name, which keeps torch out of the process -- and means slang like "I need to
pee" only resolves with `--nlu`.

`BuildingRouter` returns one leg per floor with a "take the lift" handover --
`run_building.py` is what actually drives it: A* leg, ride, A* leg, ride, A*
leg, in one continuous run. Video is `output/scene_cache/building.mp4`: chase view on
the left with a three-pip floor indicator and the names of the rooms being
passed, the dog's own 640x480 camera on the right with the projected goal pixel
(and VAMOS's candidate paths under `--vamos`). See [Reading the
video](#reading-the-video) for why the names are on the left panel only.

### Reading the video

Two captions are drawn on the chase panel, and both exist because the building
renders as one grey corridor repeated 48 m and three times over:

- **three pips down the left edge**, the current storey lit. Without them the
  video cannot tell you which floor you are on.
- **a door sign on each room the dog is passing**, from `locations.json`, out
  to 8 m and fading in at the edge of that. Without them the video cannot tell
  you *where along* the floor you are: a run to `room 201` and a run to
  `room 206` are otherwise the same footage.

  Drawn as a bordered placard at 1.55 m and **sized by distance** -- nearer is
  bigger, the way a real sign is. Both of those were corrections: at the 1.95 m
  the first version used, the sign sits at the wall/ceiling junction (walls are
  2.0 m) and reads as a caption floating over the corridor rather than
  something mounted by the door, and at a fixed size a name 8 m away competes
  with the one you are actually trying to read.

Both are drawn onto the rendered frame by `overlay.ChaseCam` / `label_places`,
**not built into the scene**. That boundary is the same one `obstacles.py` and
`pedestrians.py` keep. Door signage as geoms would look identical on the left
panel and would also appear in the right one -- the dog's camera, which is what
VAMOS is handed -- and VAMOS is a VLM, so it can read. That would hand the model
a text cue it does not have today and quietly change the experiment. The names
are a caption for the viewer; the robot is told nothing.

Names collide constantly down a corridor seen nearly end-on, so the nearest one
wins and the rest are dropped for that frame rather than drawn underneath:
overprinting turns "main entrance" and "library" into "mai library nce".

`locations.json` needs two corrections before it can be signage, because it is
a routing table: facing rooms share one `door_xy` on the centreline (`room 101`
and `room 106` are both [3.0, 9.5]), and one place carries several names for
the parser (`stairs` / `staircase` / `stairway` / `stairwell`). So a sign slides
1.25 m off the centreline towards its own room -- onto that room's wall, which
is where signage lives -- and synonyms collapse to the shortest name. Without
the first, one room of every facing pair is dropped as an overlap and half the
corridor goes unlabelled.

They are overlays, so they do not occlude: a sign 8 m ahead draws over the wall
between, rather than being hidden by it. Down a straight corridor that reads
correctly enough, and depth-testing a caption is not worth a depth buffer.

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
robot cannot press a button, so a human does. Stopping short is literal --
the router's handoff point (`lift.LIFT_XY`) is inside the car, so the leg to
the lift is cut back to `lift.WAIT_XY`, a standoff outside the open face, and
the walk in happens only after the handover is answered. `--auto-confirm` answers for
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

| floor | box | footprint | open side | A* leg that hits it |
|---|---|---|---|---|
| 1 | trolley, x 23.4-24.6 | 1.2 x 0.7 m, 0.95 m high | north, 0.82 m | `room 101`, `room 106`, `room 108` |
| 2 | crate, x 34.1-35.1 | 1.0 x 0.8 m, 0.70 m high | north, 1.02 m | `room 201`-`204`, `electrical engineering lab` |
| 2 | cart, x 16.0-16.9 | 0.9 x 0.55 m, 0.95 m high | north, 1.02 m | `room 201`, `room 206`-`208` |
| 3 | cartons, x 41.2-41.8 | 0.6 x 0.6 m, 0.90 m high | north, 1.02 m | `biology lab`, `physics lab`, `chemistry lab`, `room 301` |

They are the sizes the objects actually are. An earlier version ran each box
from a wall across the centreline -- 1.45 m deep in a 2.7 m corridor -- and the
size was quietly doing the perception layer's work: over half the hallway is not
an obstacle to be seen and gone round, it is a chicane, and getting past one
needs a planned curve this stack does not have yet. It is also not a trolley.

Every box straddles the centreline -- which `obstacles.py` asserts, because a
box parked tidily against a wall is in the corridor and never in the dog's way
-- and puts its slack into one ~1 m gap on the side the routes there use. All
four open north, because that is the side the doorways these routes turn into
are on.

That layout was chosen while `free_destination` still had the three bugs listed
under **Known limits** below, which made the dog look as though it needed a
metre of corridor to commit to anything. With those fixed it passes what it
fits through, so the arrangement is no longer load-bearing -- but it is still
the arrangement these twenty routes were measured on, so re-run them before
trusting a different one.

Floor 3 is the tightest. Those routes fan out to doors on *both* sides of the
corridor from a single lift, so whichever side the cartons leave open, routes
wanting the other side get 0.52 m. `chemistry lab` is the one that wants it: it
used to drive straight through the cartons, then stopped short of its door for a
long time, and it now arrives with 0.32 m in hand. Getting in means passing them
on the north and turning 90 degrees south into a door 1.2 m on -- still a curve
no sideways-displaced goal can express, but what was actually stopping the dog
was changing its mind about which side to pass on, tick by tick. See
`free_destination(prefer=...)`.

Re-check the table after any change to the grids or the router. `MIN_GAP` is
asserted against the wider side, so it cannot catch a box whose open side is the
wrong one -- which side is useful depends on the routes, not the geometry.

They are still absent from the map. What changed is that the dog can now see
them -- `lidar.py` and `perception.py` below -- and goes round.

## Seeing them, and going round

`lidar.py` is the virtual Mid-360 the spec asks for at L2 step 2: ~2000 rays,
360 degrees, **~1.5 ms a scan**, emitting the `/lidar/points`-shaped cloud L5 and
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

VAMOS drives at whatever goal pixel it is handed, and the destination comes from A*
on a map with no crate in it -- so the pixel lands *on* the crate. Measured
before any of this was built: at 0.8 m the crate fills 60% of the frame and all
five candidates still go straight through it. That is not a model that cannot
avoid obstacles, it is a model being told to walk into one, and rejecting all
five leaves nothing to follow.

So the map's half of "map decides WHERE, VLM decides HOW" uses the sensor:
`free_destination` moves the goal sideways into the gap. Which side is decided at
the *pinch* -- the tightest point on the way -- because judged at the destination
nothing looks blocked until the dog is level with the crate, and judged beyond
it the roomiest direction is "straight on". How far is decided at the destination,
because the shift that centres the pinch puts a destination that is already past the
crate into the corner it just cleared.

While a detour is on, that displaced goal steers and VAMOS keeps proposing and
being judged -- its candidates and the gate's rejections are both in the video.
That split is a measurement, not a demotion: zero-shot VAMOS spreads its five
candidates about **+/-0.25 m over a 2 m path**. Against the 1.45 m-deep boxes
this scene used to carry, getting round one took about **0.65 m** -- so its
paths were safe over their own length, passed the gate, and still walked the dog
into the thing, which the gate cannot catch because a path that is fine for 2 m
and leads nowhere at 4 m is a path it has no reason to reject. Realistically
sized boxes brought that sidestep inside the model's own spread, which is why
`room 201` now finishes under `--vamos` (below). The limit is unchanged, only
the corridor is: a turn wider than the horizon is still the map's to make, and
spec L4 step 5, the LoRA fine-tune, is what would move it to the model.

### People, who are not crates

`pedestrians.py` adds the other half of the experiment, and it is deliberately
the opposite of the crates in every way that matters. They are on no map
either, but they *move*, so a scan that stands alone cannot describe them --
and the right answer is not to squeeze past. A guide dog that threads a moving
gap is towing a blind person through it.

```
python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --seed 1
```

Capsule people walk the corridors of every floor, randomly but reproducibly
per seed: some crossing wall to wall, some walking its length in a side lane.
Nothing about them is told to the robot; they are geoms, and the LiDAR finds
them like anything else.

Reaching the end of a leg is a corner, not an exit. They turn and carry on down
the corridor from where they stopped, in the direction they were already
headed, and are only ever recycled somewhere else while the dog cannot see them
-- which is a distance test (12 m, the LiDAR's range), because the Mid-360 is a
360-degree sensor and "behind the dog" is not unseen. A body that vanishes off
the end of its line four metres away reads to `tracking.py` exactly as one that
appears there: as something moving very fast. Over 12 seeds x 60 s that is 165
relocations with 0 of them visible.

Three things they deliberately do not do, each of which was tried:

- *walk back down the same line.* A crosser pacing the same two metres is a
  moving wall, and a dog that waits politely for one never gets down the
  corridor.
- *stand still at the end of a leg.* A stopped person is a permanent 0.44 m
  obstacle, and in a 2.7 m corridor whose route hugs one wall the dog can only
  squeeze past -- measured at 0.09 m, which is through them, the capsules being
  `contype=0`. It also silences the yielding: a standing person is not a mover,
  so `conflict()` stops firing and the dog waits for nobody.
- *turn towards the middle of the corridor* -- which is what picking the new
  direction from their position rather than their heading does. It funnels the
  whole crowd into the centre to pace there, the moving wall this module exists
  not to build, and it livelocked the floor-2 restroom route for 12 minutes of
  wall clock against 41 seconds of sim time.

Five seeds of the floor-1 restroom route, three people, scored on ground truth:

```
                contacts   seeds that waited
    vanishing     0/5            4/5
    turning       0/5            4/5
```

Turning costs nothing measurable against the version that vanished -- same clean
contact record, same yielding -- and buys a crowd that no longer teleports on
camera. The standing variant, measured when it was tried, scored 2 contacts in
those five seeds (worst 0.09 m) and waited in 1, and spent 65% of walker-ticks
stationary against 19% for turning.

`tracking.py` is what makes them different from a post. It clusters the
unexplained returns, matches the clusters to last tick's, and measures
velocity **across a half-second window rather than between two ticks** -- which
is the whole reason it works. A stationary crate is not a stable object to a
range sensor: its visible face grows, shrinks and occasionally breaks into two
clusters, and the centroid jumps half a metre in a tick when it does. As a
tick-to-tick difference that is 10 m/s, and smoothing it just spreads one jump
across the walking range. What a crate cannot do is *travel*, so half a second
of rattle nets out to nothing while half a second of walking is over half a
metre.

Two shape filters do the rest, and between them they took the false stops in
an empty building from 19 to zero:

- a cluster of one or two returns is a ray grazing a corner, and its position
  is wherever that ray landed;
- a cluster wider than 0.5 m is not one person. The crates come back at
  0.55-0.90 m and their centroid slides along that face at 0.6 m/s as the dog
  walks past, which is a walking pace -- a speed threshold alone cannot
  separate them, but shape can.

The decision is then **closest approach**, not proximity: do the two straight
lines meet inside the next 2.5 s. Near is the wrong test in both directions --
someone walking away two metres ahead is near and irrelevant, someone crossing
four metres ahead at 1.4 m/s is far and about to be exactly where the dog will
be. A crossing four metres out is correctly *not* a stop: 2.1 m of corridor at
walking pace takes under two seconds, by which time they are at the far wall.

Two things keep it from dithering. Starting to wait asks 0.95 m; carrying on
waiting asks 1.30 m, so the dog makes one clean stop and one clean start
instead of shuffling past somebody. And it waits for *that person* until they
are clear, moving or not -- a person who has stopped because the dog is in
front of them has no velocity, so a moving-only test calls them a crate and
drives at them, and they stay stopped. After three seconds of genuine
stillness the latch releases and they become `free_destination`'s problem, to be
gone round like any other obstacle.

The two are kept apart in the costmap as well: a walking person is taken out
of the field `free_destination` plans detours around, while staying in the one the
safety gate and the proximity stop read. The gap beside somebody is a gap that
is leaving, and a route committed to it is committed to where they were.

Scored on ground truth the robot never sees, six seeds of three people on the
floor-2 route:

```
python -m cyberdog.sim.run_building "room 201" --pedestrians 3 --seed 1 --no-video
  ARRIVED on floor 2 at (3.0, 4.5) after 102s of sim time
  collisions: none -- the dog never entered an obstacle's footprint
  people: 6 times it stopped to let someone past, 16.7s waiting in total
  contact: none -- closest it came to anybody was 0.63 m
```

**6/6 arrive, none of them touching a crate or a person.** The closest anybody
came was 0.55 m -- the threshold itself -- and four of the six stayed beyond
0.6 m.

That is the first zero this project has measured, and it is worth saying what
produced it, because it was not politeness and it was not the scene. The
documented figure was 2 of 6 seeds inside the threshold; resizing the obstacles
made it 3 of 6. Both numbers had the same cause: `free_destination` measured its
sideways crossing at the obstacle but *applied* it at the destination, so the
dog drove a shallow diagonal and drew level with a crate still half-way across,
finishing the squeeze right where `SIDE_LANES` puts a walking lane. Crossing by
the time it reaches the obstacle -- see **Known limits** -- moved every one of
those margins the right side of the line. The stopping logic below is unchanged.

With the yielding disabled and everything else identical, the same seeds produce
contact throughout -- which is the measurement that says the stopping is doing
the work, rather than the crowd happening to miss.

`restroom` (floor 1) with `--seed 5` used to be the one route/seed that did not
arrive: a walker stopped in open corridor, the latch released them as an
obstacle after three seconds, and the dog reported "I cannot find a way past
this" at (41.0, 8.6). It arrives now, in 43 s. Nothing about the people changed
-- it was the same wall-reasoning bug as everything else in **Known limits**:
the destination two metres past a stationary person is often around the corner
into their room, and the line to it goes through the wall.

### When there is no way past

Three steps, not one rule. Crawl at 30% while re-asking VAMOS sooner than the
usual 1 Hz; stop when the dog is touching distance from something with nothing
approved; and after 8 s of that, say so and end the leg rather than grinding on
in silence.

Stopping is not on its own a recovery, and for a long time it was treated as
one: clearance cannot improve while the dog does not move, so every graze ran
the full 8 s and ended the leg. Squeezing past the floor-2 cart leaves 0.18 m
against a 0.16 m threshold -- the empty building clears it and the same route
with people somewhere else in it did not, purely because they perturbed the
dog's line by two centimetres. So the proximity stop now backs off at 0.15 m/s
the way the dog came, after checking that it is still clear back there. Two
centimetres should cost a step backwards, not a failed run. Announcements come *before* the lean, not during -- the person's
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
not at all -- and prints the result.

Map-only, no `--vamos`:

```
python -m cyberdog.sim.run_building "room 201" --auto-confirm --no-video
  ARRIVED on floor 2 at (3.1, 4.5) after 83s of sim time
  obstacles: up to 328 returns the map could not explain, 0.0s crawling, 0.0s stopped, least clearance ahead 0.11 m
  collisions: none -- the dog never entered an obstacle's footprint
```

Across every named destination in the building, map-only: **they all arrive,
none of them having touched anything** -- `chemistry lab` and the floor-2
`restroom` included, which is new (see `prefer`, below). The
six-route suite in `selftest.py` missed a collision for three successive layouts
of this scene, because none of its routes came at a box from the side the slack
was not on; it now runs eight, including both sides of the cartons, and scores
each against what it is known to do.

**With `--vamos`, this route now finishes.** Two consecutive runs both arrived,
in 94 s against the map-only 83 s:

```
python -m cyberdog.sim.run_building "room 201" --vamos --auto-confirm --no-video
  ARRIVED on floor 2 at (3.0, 4.5) after 92s of sim time
  VAMOS floor 1: 12 calls, 0 failed, 55 candidates rejected (17 of them by the safety gate), mean safety of the paths it followed 0.87
  VAMOS floor 2: 72 calls, 0 failed, 95 candidates rejected (15 of them by the safety gate), mean safety of the paths it followed 1.00
  obstacles: up to 356 returns the map could not explain, 13.6s crawling, 0.0s stopped, least clearance ahead 0.45 m
  collisions: none -- the dog never entered an obstacle's footprint
```

That is a change of outcome, and the boxes are the whole reason for it. Against
the 1.45 m-deep ones, four consecutive runs all ended at the floor-2 cart, with
the safety gate rejecting 113 of 149 candidates -- VAMOS's +/-0.25 m spread over a
2 m path against the 0.65 m of sidestep that cart demanded. The cart is now
0.55 m deep with a metre of corridor beside it, the sidestep is inside what the
model actually proposes, and the gate rejects 15 of 119 instead of 113 of 149.
Nothing about the VLM side changed: same server, same prompt, same gate.

`electrical engineering lab` behaves the same way, and it is the better
measurement of the two because it turns into a doorway at the end: 42 calls,
146 rejected, none of them by the gate, arrives in 54 s. With three people in
the corridor it still arrives (57 s); on the code before the `free_destination`
fixes that same run failed at (37.6, 10.1). The mean safety of 1.00 on
`room 201` above is the same effect from the other side -- the VLM is no longer
being asked to draw paths towards goals that sit behind walls, so fewer of its
proposals are unusable before the gate ever sees them.

What has *not* changed is the shape of the limitation. The gate is still doing
real work, the model still proposes short paths that are safe over their own
length, and where a box demands more sidestep than +/-0.25 m the map still has to
make the turn (`perception.free_destination`). This route no longer demands it.
Treat "arrives under `--vamos`" as a statement about this corridor, not about
zero-shot VAMOS: **0 failed calls** in either run is the part that generalises.

It is also not reproducible from a seed. The VLM samples at `temperature=1.0`
and `vamos_client.py` sends no temperature, so two identical commands give
different paths; the two runs above differed by 2 calls and 16 candidates.

The floor-1 numbers in that block are small for a reason that is not the VLM's:
`main entrance` (45, 4) is 1.6 m from the lift, so the floor-1 leg of any
cross-floor run is about five metres of turning out of a doorway. VAMOS is
called a dozen times there and nearly every candidate it draws goes through the
entrance wall, which is why a chosen path is on screen for about a second in
`building.mp4`. Floor 1 is a lobby on these routes, not a corridor; to watch the
VLM steer down one, give it a floor-1 destination (`room 101`, `cafeteria`).

`selftest.py` does not use `--vamos`, so `ALL PASS` says nothing about the VLM
path -- the block above is the only score it has.

## Known limits

- The dog is kinematic -- no contacts, and the gait is drawn, not simulated.
- The obstacles are seen, but there is no memory: each scan stands alone, which
  is fine for a 360-degree sensor and wrong the moment something is occluded.
  Beyond LiDAR range an imagined rollout is still scored on the static map.
- CE-RRT* (L6 §2) does not exist -- VAMOS proposes and the gate disposes.
- **A detour holds its side for as long as the obstacle is in sight, and
  nothing plans the curve past that.** Which side has more room is judged
  afresh every tick, from a scan that changes as the dog closes in, and level
  with an obstacle the answer alternates: measured beside the floor-2 crate, it
  flipped between 0.7 m north of the pinch and 0.9 m south of it, tick about.
  The dog leant north, was sent south the next tick, and drove into the crate it
  was going round; `chemistry lab` stopped short for the same reason.
  `free_destination(prefer=...)` commits to the first side chosen and only
  reconsiders when that side has nothing reachable left, which is what those two
  routes were waiting for. A genuine curve -- round an obstacle and immediately
  through a door -- is still CE-RRT* (L6 §2), and still does not exist.

Three entries that used to be on this list are gone, and all three were the same
kind of mistake -- a test that asked a question other than the one that
mattered. Kept here because the symptoms were blamed on thresholds for a while,
and they were not:

- **Avoidance ran only while the camera could see the goal pixel.** The obstacle
  path in `run_building.follow` sat behind `if state["state"] == "TRACK"`, and
  `ALIGN` is exactly what the projector reports while the dog turns into a
  doorway -- so for the whole of every turn it steered at the raw A* waypoint
  with the LiDAR ignored. That was the `chemistry lab` collision: 0.4 s inside
  the cartons, every tick of it in `ALIGN`.
- **Straight lines were drawn through walls.** Within 2 m of a corner,
  `pick_destination` interpolates past it, so `free_destination` was handed a
  goal metres inside a room. The pinch on that line is the wall, every offset
  from it is inside the building, and the answer is "no way past" -- measured
  with 0.68 m of clear floor on every side of the dog. `in_open_floor` now
  clamps the geometry to the part of the line that is in the building.
- **The crossing finished at the destination, not at the obstacle.** Told to
  cross 0.70 m, the dog reached the cartons with 0.24 m in hand against the
  0.20 m it insists on. This is what produced the pedestrian grazes.

None of the thresholds moved. `DESTINATION_CLEAR`, `LINE_NEED` and `STATIC_MIN`
were measured first, and none of them was binding in any of the three failures.
- `--vamos` completed `room 201` on the current geometry (2 runs for 2, 94 s
  against 83 s map-only) and could not on the 1.45 m boxes (0 runs for 4). Its
  +/-0.25 m spread makes it a measurement of the corridor as much as of the
  model: a turn wider than its 2 m horizon is still the map's to make, and spec
  L4 s5, the LoRA fine-tune, is what would change that.
- `--vamos` is **not reproducible**: the server samples at `temperature=1.0` and
  the client sends no temperature or seed, so two identical commands differ.
  `--seed` fixes the crowd, not the VLM.
- The people are capsules on scripted legs, not a pedestrian model. They know
  exactly one thing about the dog -- do not walk into the thing in front of
  you -- and making them any cleverer would quietly solve the robot's problem
  for it. They do not step around it, and they will walk into its side.
- The dog stops for a person rather than flowing around one. The spec (L6
  acceptance) wants a pedestrian avoided *without* a full stop, which needs a
  planner that can commit to a curve; until CE-RRT* exists, stopping is the
  honest version and the full stops are counted in every run's summary.
- Tracking has no occlusion model and matches by nearest centroid, so someone
  who steps behind a crate is a lost track that has to earn `moving` again,
  and two people passing each other can swap identities.
- The lift has no doors and no call delay: the car is always where the dog is.
- The Gemma layer was fine-tuned before the lift existed, so `--nlu` cannot
  parse "take me to the lift" as a destination -- it maps unknown words onto
  the nearest name it knows. That is why hazard names are checked against the
  map with rules *before* the model runs (`BuildingRouter.hazard_named`), and
  why "take me to the stairs" (or staircase / stairway / stairwell) walks to
  the corridor outside them and says it goes no closer, on either path. Add
  `elevator`/`lift` to `cyberdog/language/generate_dataset.py` on the next
  regeneration.
