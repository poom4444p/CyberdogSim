"""Collect affordance trials in Isaac Lab: what can the Go2 actually walk over?

Spec Layer 5, step 2. Thousands of Go2s on Isaac Lab's rough terrain (stairs
up and down, boxes, rough ground, slopes), each walked by a trained
locomotion policy. Every dog repeatedly:

    1. looks at the ground around it   -> an elevation patch (affordance/data.py)
    2. is given a target ~1 m ahead     -> drives at it with the TWIN's control
                                           law (K_W, TURN_ONLY, 0.8 m/s), the
                                           same one dreaming.py imagines
    3. arrives, falls, or runs out of time -> one record

Records are raw (arrived? fell? how far did the body tilt and rise?); what
counts as walkable is decided later by data.label(), on the Mac.

Runs only inside Isaac Lab, on an NVIDIA GPU. From the IsaacLab folder:

    ./isaaclab.sh -p /path/to/Cyberdog/scripts/isaac/collect_affordance.py \\
        --policy /path/to/exported/policy.pt --headless

`policy.pt` is the TorchScript file play.py exports next to the checkpoint
(scripts/isaac/README.md, step 2). Output: one folder per run under --out,
a .npz shard every --shard_size trials, so a run cut short keeps what it had.
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--policy", required=True,
                    help="TorchScript policy.pt exported by play.py")
parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--trials", type=int, default=50_000,
                    help="stop after this many records")
parser.add_argument("--shard_size", type=int, default=5_000)
parser.add_argument("--out", default=None,
                    help="output folder (default: <repo>/datasets/affordance)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim has to be running before anything else from isaaclab is imported.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import datetime  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  registers the tasks with gym
from isaaclab.utils.math import wrap_to_pi  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# The repo does not need to be pip-installed into Isaac's Python: put its
# source on the path. The control law comes from the twin itself, so the
# dogs here drive the way the dog there does.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cyberdog.affordance import data  # noqa: E402
from cyberdog.sim.control import K_W, TURN_ONLY  # noqa: E402
from cyberdog.sim.sensing.dreaming import (ARRIVED_R, HORIZON_SLACK,  # noqa: E402
                                           TURN_ALLOWANCE_S)

SETTLE_S = 1.0          # after a reset, stand this long before the first trial
EPISODE_S = 60.0        # Play's 20 s throws away a trial in progress too often
SCAN = 3.2              # the wider scanner: a centred square that covers the patch
LOG_EVERY_S = 10.0


def make_env():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = EPISODE_S

    # The commands are ours: no resampling, no heading mode, nobody told to
    # stand still. Written into the command term every step (see main()).
    vc = cfg.commands.base_velocity
    vc.heading_command = False
    vc.rel_standing_envs = 0.0
    vc.resampling_time_range = (1e9, 1e9)
    vc.debug_vis = False

    # Terrain laid out the training way -- a column per terrain type, a row per
    # difficulty -- so a record's tile says what it was walking on. No
    # curriculum moving dogs between rows: spawn rows stay uniform.
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols, gen.curriculum = 10, 20, True
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None

    # A second, wider height scanner for the patch. The policy's own scanner
    # (1.6 x 1.0 m) is part of its observation, so it stays exactly as it was.
    hs = cfg.scene.height_scanner
    cfg.scene.affordance_scanner = hs.replace(
        pattern_cfg=hs.pattern_cfg.replace(size=[SCAN, SCAN]), debug_vis=False)

    return gym.make(args.task, cfg=cfg), cfg


def column_names(gen):
    """Sub-terrain name of each column, assigned the way TerrainGenerator does."""
    names = list(gen.sub_terrains)
    props = np.array([gen.sub_terrains[n].proportion for n in names], dtype=float)
    cum = np.cumsum(props / props.sum())
    return [names[int(np.argmax(j / gen.num_cols + 0.001 < cum))]
            for j in range(gen.num_cols)]


def main():
    env, cfg = make_env()
    uenv = env.unwrapped
    dev, N, dt = uenv.device, uenv.num_envs, uenv.step_dt
    robot = uenv.scene["robot"]
    scanner = uenv.scene.sensors["affordance_scanner"]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    w_lo, w_hi = cfg.commands.base_velocity.ranges.ang_vel_z

    policy = torch.jit.load(args.policy, map_location=dev).eval()
    rng = np.random.default_rng(args.seed)

    # Tile centres, to say which tile a trial started on: dogs wander off the
    # tile they spawned on within a few trials.
    origins = uenv.scene.terrain.terrain_origins          # (rows, cols, 3)
    n_cols = origins.shape[1]
    origins_xy = origins[..., :2].reshape(-1, 2)

    out = Path(args.out) if args.out else REPO / "datasets" / "affordance"
    run_dir = out / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    meta = {
        "task": args.task, "policy": str(Path(args.policy).resolve()),
        "num_envs": N, "seed": args.seed, "step_dt": dt,
        "patch": {"res": data.RES, "x0": data.X0, "nx": data.NX,
                  "y0": data.Y0, "ny": data.NY, "scan": SCAN},
        "control": {"walk_v": data.WALK_V, "k_w": K_W, "turn_only": TURN_ONLY,
                    "w_range": [w_lo, w_hi], "arrived_r": ARRIVED_R},
        "target_d": list(data.TARGET_D),
        "target_bearing": float(data.TARGET_BEARING),
        "terrain_columns": column_names(cfg.scene.terrain.terrain_generator),
    }

    # Per-dog trial state.
    active = torch.zeros(N, dtype=torch.bool, device=dev)
    since_reset = torch.zeros(N, dtype=torch.long, device=dev)
    steps = torch.zeros(N, dtype=torch.long, device=dev)
    limit = torch.zeros(N, dtype=torch.long, device=dev)
    target_w = torch.zeros(N, 2, device=dev)
    start_z = torch.zeros(N, device=dev)
    last_z = torch.zeros(N, device=dev)
    z_lo = torch.zeros(N, device=dev)
    z_hi = torch.zeros(N, device=dev)
    max_tilt = torch.zeros(N, device=dev)
    start_patch = np.zeros((N, data.NY, data.NX), dtype=np.float32)
    start_target = np.zeros((N, 2), dtype=np.float32)
    start_tile = np.zeros((N, 2), dtype=np.int16)

    buffer = {k: [] for k in data.FIELDS}
    totals = {"n": 0, "reached": 0, "fell": 0, "shards": 0}

    def flush():
        if not buffer["patch"]:
            return
        records = {k: np.stack(v) if k in ("patch", "target") else np.array(v)
                   for k, v in buffer.items()}
        records = {k: records[k].astype(_dtype(k)) for k in records}
        path = run_dir / f"shard_{totals['shards']:04d}.npz"
        data.save_shard(path, records, meta)
        print(f"[collect] wrote {len(records['patch'])} trials to {path}")
        totals["shards"] += 1
        for v in buffer.values():
            v.clear()

    def finish(mask, reached, fell):
        ids = mask.nonzero().flatten().tolist()
        for i in ids:
            buffer["patch"].append(start_patch[i].copy())
            buffer["target"].append(start_target[i].copy())
            buffer["reached"].append(reached)
            buffer["fell"].append(fell)
            buffer["duration"].append(steps[i].item() * dt)
            buffer["max_tilt"].append(max_tilt[i].item())
            buffer["dz"].append(last_z[i].item() - start_z[i].item())
            buffer["z_span"].append(z_hi[i].item() - z_lo[i].item())
            buffer["terrain_level"].append(start_tile[i, 0])
            buffer["terrain_type"].append(start_tile[i, 1])
        active[mask] = False
        totals["n"] += len(ids)
        totals["reached"] += len(ids) if reached else 0
        totals["fell"] += len(ids) if fell else 0
        if len(buffer["patch"]) >= args.shard_size:
            flush()

    def start(mask):
        ids = mask.nonzero().flatten()
        if not len(ids):
            return
        pos = robot.data.root_pos_w[ids]
        yaw = robot.data.heading_w[ids]
        poses = torch.stack([pos[:, 0], pos[:, 1], yaw], dim=1).cpu().numpy()
        pts = scanner.data.ray_hits_w[ids].cpu().numpy()
        gz = data.ground_under(pts, poses)
        seen = np.isfinite(gz)               # nothing under it: try next step
        if not seen.any():
            return
        keep = torch.as_tensor(seen, device=dev)
        ids, pos = ids[keep], pos[keep]
        poses, pts, gz = poses[seen], pts[seen], gz[seen]
        n = len(ids)

        d = rng.uniform(*data.TARGET_D, n)
        b = rng.uniform(-data.TARGET_BEARING, data.TARGET_BEARING, n)
        tb = np.stack([d * np.cos(b), d * np.sin(b)], axis=1).astype(np.float32)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        tw = poses[:, :2] + np.stack([c * tb[:, 0] - s * tb[:, 1],
                                      s * tb[:, 0] + c * tb[:, 1]], axis=1)

        cpu = ids.cpu().numpy()
        start_patch[cpu] = data.patches_from_points(pts, poses, gz)
        start_target[cpu] = tb
        tile = torch.cdist(pos[:, :2], origins_xy).argmin(dim=1).cpu().numpy()
        start_tile[cpu, 0], start_tile[cpu, 1] = tile // n_cols, tile % n_cols

        target_w[ids] = torch.as_tensor(tw, dtype=torch.float32, device=dev)
        horizon = TURN_ALLOWANCE_S + HORIZON_SLACK * d / data.WALK_V
        limit[ids] = torch.as_tensor(np.ceil(horizon / dt), dtype=torch.long, device=dev)
        steps[ids] = 0
        z = pos[:, 2]
        start_z[ids], last_z[ids], z_lo[ids], z_hi[ids] = z, z, z, z
        max_tilt[ids] = 0.0
        active[ids] = True

    env.reset(seed=args.seed)
    settle = int(round(SETTLE_S / dt))
    t0 = t_log = time.time()
    first = True

    with torch.inference_mode():
        while simulation_app.is_running() and totals["n"] < args.trials:
            start(~active & (since_reset >= settle))

            # The twin's control law, vectorised: pivot when badly off, else walk.
            p = robot.data.root_pos_w[:, :2]
            to = target_w - p
            err = wrap_to_pi(torch.atan2(to[:, 1], to[:, 0]) - robot.data.heading_w)
            v = torch.where(err.abs() > TURN_ONLY, torch.zeros_like(err),
                            data.WALK_V * torch.cos(err))
            w = torch.clamp(K_W * err, w_lo, w_hi)   # the policy's trained range
            cmd = torch.stack([v, torch.zeros_like(v), w], dim=1)
            cmd_term.vel_command_b[:] = cmd * active.unsqueeze(1)

            # Observe again, so the policy sees this step's command, not last's.
            obs = uenv.observation_manager.compute()["policy"]
            if first:
                _check_policy(policy, obs)
                first = False
            _, _, terminated, truncated, _ = env.step(policy(obs))

            # Done envs have already been reset inside step(). A fall is a
            # result; a time-out is not -- the trial just never got to finish.
            finish(active & terminated, reached=False, fell=True)
            active &= ~truncated
            done = terminated | truncated
            since_reset = torch.where(done, torch.zeros_like(since_reset), since_reset + 1)

            z = robot.data.root_pos_w[:, 2]
            g = robot.data.projected_gravity_b[:, 2]
            tilt = torch.acos(torch.clamp(-g, -1.0, 1.0))
            last_z = torch.where(active, z, last_z)
            z_lo = torch.where(active, torch.minimum(z_lo, z), z_lo)
            z_hi = torch.where(active, torch.maximum(z_hi, z), z_hi)
            max_tilt = torch.where(active, torch.maximum(max_tilt, tilt), max_tilt)
            steps += active.long()

            dist = torch.linalg.norm(target_w - robot.data.root_pos_w[:, :2], dim=1)
            arrived = active & (dist < ARRIVED_R)
            finish(arrived, reached=True, fell=False)
            finish(active & (steps >= limit), reached=False, fell=False)

            if time.time() - t_log > LOG_EVERY_S:
                t_log = time.time()
                n = max(totals["n"], 1)
                print(f"[collect] {totals['n']}/{args.trials} trials, "
                      f"{totals['n'] / (t_log - t0):.0f}/s, "
                      f"reached {totals['reached'] / n:.0%}, fell {totals['fell'] / n:.0%}")

    flush()
    print(f"[collect] done: {totals['n']} trials in {run_dir}")
    env.close()


def _dtype(field):
    return {"reached": bool, "fell": bool, "terrain_type": np.int16,
            "terrain_level": np.int16}.get(field, np.float32)


def _check_policy(policy, obs):
    """Fail on the first step, in words, if the policy is for another task."""
    try:
        policy(obs)
    except RuntimeError as e:
        raise SystemExit(
            f"[collect] the policy does not take this task's observation "
            f"({obs.shape[1]} values). Was it trained on {args.task.replace('-Play', '')}?\n{e}")


if __name__ == "__main__":
    main()
    simulation_app.close()
