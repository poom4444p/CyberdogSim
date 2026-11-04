# Hardware in the loop: the J4012 as the dog

Everything the dog would carry runs on the J4012; the world it walks in runs
on the Mac. They talk over the USB-C link, in the messages a real Go2 speaks.

| Where | What | How it starts |
|---|---|---|
| J4012, cores 6-7 | go2_board `board_main`: motors (DDS), safety, the walking policy, the Mid-360 guard | `onboard.sh board` |
| J4012, cores 0-5 | the Cyberdog brain (`run_building --robot go2_board`) | `onboard.sh brain DEST` |
| J4012, GPU | VAMOS | `onboard.sh vamos` |
| Mac | go2_world: physics, people, the lift (`run_world.py`); the Mid-360 (`run_lidar.py`) | `run_hil.sh` |

## Files

| File | Runs on | |
|---|---|---|
| `run_hil.sh` | Mac | a whole run: world, LiDAR, board, brain; logs and a summary |
| `onboard.sh` | J4012 | starts and stops the board's side; `run_hil.sh` calls it over ssh |
| `trace_run_building.py` | J4012 | the brain, printing its commands (`TRACE=1`) |
| `dds_gaps.py` | either | how regularly states and commands arrive on the link |

## A run

```
bash scripts/hil/run_hil.sh sync                 # after changing Cyberdog or go2_board
bash scripts/hil/run_hil.sh clock                # after the board boots (asks for its sudo password)
PY=/opt/anaconda3/envs/unitree_sim/bin/python bash scripts/hil/run_hil.sh walk
PY=/opt/anaconda3/envs/unitree_sim/bin/python bash scripts/hil/run_hil.sh route "server room"
```

`PY` is a Python with go2_world's packages (mujoco, unitree_sdk2py). Logs go to
`output/hil/<time>-<walk|route>/`.

`sync` copies the two repos **as checked out**: it prints each one's branch and
commit and asks before copying (`SYNC_YES=1` when there is no terminal). Check
the go2_board line: a sync from a checkout without the board fixes puts the old
board on the J4012, and only a fall says so.

- **`walk`** (about a minute): the board stands, walks 4 s at 0.54 m/s and
  0.47 rad/s, and stands again, driven by go2_board's `brain_cli.py`, no brain.
  It ends `RESULT: walked` or `RESULT: FELL`. Run it first after any change to
  go2_board: it is what showed the board steering on states 0.3 s old.
- **`route DEST`** (3-4 minutes): the whole route with the brain and VAMOS, e.g.
  floor 1 to the server room on floor 3, with the lift and four people.

Both end with what `dds_gaps.py` saw on the board's side of the link:

```
lowstate (world)    500/s  median  2.0  p99   2.6  ...
lowcmd (board)      466/s  median  2.0  p99   3.5  ...
```

`lowcmd` is the board loop's rate (target 500/s). A separate process with its
own socket, it shows what is on the wire, not how fresh the states the board
uses are: those need go2_board's own check (`tests/test_sim_bus.py`).

## Settings worth knowing

- **`STALE_MS`** (`onboard.sh`, default 200): the board goes limp after this long
  without a state. The Mac's world now and then pauses for 100 ms and more, so
  the rig uses 200. go2_board's own limit is 50 ms (`safety.STALE_S`); set
  `STALE_MS=` (empty) to use it. Anything driving real motors must.
- **`WORLD_ARGS=--held-pd`**: go2_world's motors as unitree_mujoco has them (the
  torque worked out once per command and held) instead of every physics step.
- **Cores**: `BOARD_CORES` (6,7) and `BRAIN_CORES` (0-5) in `onboard.sh`.

## The board's clock

The J4012 has no working RTC (`timedatectl`: RTC time 1970). After a boot its
clock is wherever it was at the last shutdown, until NTP sets it -- and with no
network but the USB-C link, nothing does. The rig runs on monotonic clocks, so
a wrong clock spoils only log times; `run_hil.sh` warns when it is more than
2 s off, and `run_hil.sh clock` sets it from the Mac (sudo on the board, so it
asks for the password). To set it without a password, a sudoers rule on the
board can allow `date` alone, e.g. with `sudo visudo -f /etc/sudoers.d/date`:

```
jittarin ALL=(root) NOPASSWD: /usr/bin/date
```

That is a decision for whoever runs the board; the scripts do not make it.
