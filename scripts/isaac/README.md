# Isaac Lab: collecting affordance data

Spec Layer 5, steps 1–2: find out what the Go2 can actually walk over, so the
affordance MLP can learn it. Everything in this folder runs **only inside
Isaac Lab on an NVIDIA GPU** (a Linux machine, e.g. SDU UCloud with a GPU
allocation). It does not run on a Mac or on a CPU-only machine.

> **Status: written, not yet run.** `collect_affordance.py` was checked against
> the Isaac Lab and rsl_rl sources (Oct 2026) but has not been run on a GPU yet.
> Expect to fix a thing or two the first time. The data format it writes
> (`src/cyberdog/affordance/data.py`) *is* tested: `pytest tests/test_affordance_data.py`.

## What comes out

One record per *trial*: a Go2 stands somewhere, is given a target 0.6–1.4 m
ahead, and drives at it with the twin's own control law (`sim/control.py`
gains, 0.8 m/s). Each record has:

- the **elevation patch** it saw at the start (21 × 21 cells, 0.1 m, +x ahead),
- the **target** in the dog's frame,
- the **raw outcome**: arrived, fell, largest tilt, how much the body rose or dropped.

What counts as *walkable* is not decided here. `data.label()` decides it on the
Mac, so a threshold can change without a GPU run. That matters for stairs: a
good policy climbs them without falling, and the label still has to say no.

## On a GPU machine, step by step

All commands below run on the GPU machine. Paths assume an SDU UCloud job with
the work folder mounted at `/work/isaac`; change them for anywhere else.

### 0. Install Isaac Lab

Follow the official pip install guide (isaac-sim.github.io/IsaacLab →
Installation → *Isaac Sim pip package*). Install everything **under `/work`**
so it survives the end of the job. Then put this repo next to it:

```bash
cd /work/isaac
git clone <this repo's URL> Cyberdog
```

The collector imports `cyberdog` from `Cyberdog/src` directly; nothing needs
to be pip-installed. It needs `numpy`, `scipy`, `PyYAML` and `Pillow` in Isaac's
Python, which Isaac Sim normally brings with it. If an import fails, run
`pip install scipy pyyaml pillow` inside the Isaac Lab environment.

### 1. Get a Go2 walking policy

Either use NVIDIA's pre-trained one (fast), or train your own (a few hours):

```bash
cd /work/isaac/IsaacLab
# pre-trained (if NVIDIA publishes one for this task):
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
    --use_pretrained_checkpoint --num_envs 16 --headless
# or train, then play the result:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 --num_envs 16 --headless
```

### 2. Find the exported policy

`play.py` writes a TorchScript copy next to the checkpoint it loaded, in a
folder called `exported/`. Its log line `Loading model checkpoint from: ...`
says where that checkpoint is. Stop `play.py` with Ctrl+C once it has printed
that line, then:

```bash
find /work/isaac -path "*exported/policy.pt"
```

### 3. A small test run first

```bash
cd /work/isaac/IsaacLab
./isaaclab.sh -p /work/isaac/Cyberdog/scripts/isaac/collect_affordance.py \
    --policy <path to exported/policy.pt> \
    --num_envs 64 --trials 500 --shard_size 250 --headless
```

Every 10 s it prints progress: trials collected, trials per second, the share
that arrived, and the share that fell. Check that:

- **reached** is well above zero. If nearly nothing arrives, the commands are
  not getting through to the policy.
- **fell** is not close to 100%. If it is, the policy is from a different task.
- two `.npz` files appear in `Cyberdog/datasets/affordance/run_<time>/`.

### 4. The real run

```bash
tmux new -s collect
./isaaclab.sh -p /work/isaac/Cyberdog/scripts/isaac/collect_affordance.py \
    --policy <path to exported/policy.pt> \
    --num_envs 512 --trials 100000 --headless
```

Detach with `Ctrl+B` then `D`; reattach with `tmux attach -t collect`. A shard is
written every 5000 trials, so if the job's time runs out you keep everything
up to the last shard.

### 5. Bring it back

Download the `run_<time>/` folder (UCloud: Files → Download) into
`datasets/affordance/` in your local checkout. On the Mac:

```python
from cyberdog.affordance import data
records, metas = data.load_shards("datasets/affordance/run_<time>")
y = data.label(records)
print(len(y), y.mean())          # how many trials, what share is walkable
```

## Options

| Flag | Default | What |
|---|---|---|
| `--policy` | (required) | the exported `policy.pt` |
| `--task` | `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` | the policy must have been trained on this task without `-Play` |
| `--num_envs` | 512 | dogs in parallel |
| `--trials` | 50000 | stop after this many records |
| `--shard_size` | 5000 | records per `.npz` file |
| `--out` | `<repo>/datasets/affordance` | where run folders go |
| `--seed` | 0 | terrain, spawns and targets |

Plus Isaac Lab's own flags, e.g. `--headless` and `--device`.

## Known gaps

- **Dense vs. sparse.** Isaac's scanner sees every cell. The twin's LiDAR
  (`sim/sensing/lidar.py`) sees far fewer and cannot see under the dog. The
  patch format allows for this (NaN means unseen, and `ground_z` is passed in),
  but an MLP trained only on dense patches will meet sparse ones at runtime.
  The fix belongs in training: drop cells at random to match the LiDAR's
  coverage.
- **Turn rate.** The Isaac policy was trained on turn rates up to ±1 rad/s, so
  the collector clamps there. The twin allows 1.5 rad/s, so the dogs in Isaac
  pivot a little slower than the dog in the twin.
- **Terrain tile.** `terrain_type` and `terrain_level` are the tile the trial
  *started* on. A target near a tile edge can be on the next tile.
