# Known limits and how they were found

The detailed history behind the short summary in the [README](../README.md#what-is-and-isnt-verified):
what was verified, the bugs that hid behind passing tests, and what is
deliberately not yet closed. Some figures are from the old defaults
(`--crowd oblivious --planner map`, before 2026-10-15); each says so where it applies.

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
  makes it (`perception.free_destination`). The LoRA fine-tune of spec L4 s5
  was tried on the twin's frames and failed Gate D, so it is deferred to the
  hardware phase. **`--vamos` can still perform worse than without it** on a floor
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
  also the dog's lane (y ≈ 10.10), and the oblivious crowd does not overtake
  on the left -- and the person against a wall on a turn round somebody.
  These figures are the oblivious crowd with the dog waiting
  (`--crowd oblivious --planner map`), the defaults before 2026-10-15.
- Perception has **little memory**. A crate seen standing is remembered for
  6 s once it is behind the dog (`perception.MEMORY_S`), because the
  nose-down Mid-360 no longer sees it and the person on the handle still has
  to pass it. Tracking keeps half a second of people, enough for a velocity
  and not enough to survive an occlusion: someone who steps behind a crate is
  a lost track.
- By default the dog **goes round** people (`--planner cerrt`, spec L6 s2,
  `planning/ce_rrt.py`) on a path planned where they will be, which halves
  the waiting, but it still stops where no path keeps P(contact) < 1%: the
  tracker's straight line is out by 1.3 m a second ahead, and a 2.5 m
  corridor often has no room for that. Every run counts its stops, and why
  each one was a stop. `--planner map` waits for everybody, as before.
- The pedestrians are capsules on scripted legs, not a pedestrian model. They
  know one thing about the dog — do not walk into what is in front of you — and
  making them cleverer would quietly solve the robot's problem for it.
- An obstacle parked against a wall is **invisible**: it falls inside the wall's
  own inflation radius and is discarded as already-explained.
