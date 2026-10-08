"""Fine-tune VAMOS with a LoRA adapter (spec L4 step 5).

The VAMOS repo ships the model and no training code, so this is the
reimplementation the spec asks for, on the data scripts/vamos/collect.py
made: (frame, "Navigate to x=<loc>, y=<loc>.") -> the 10 location tokens of
the path the dog walked.

    conda activate vamos_mac                     # torch lives here, not in cyberdog_sim
    python scripts/vamos/train_lora.py           # -> models/vamos_lora/
    python scripts/vamos/train_lora.py --epochs 1 --limit 200    # a quick look

Config per the spec: r=16, alpha=16, dropout 0.05 on q,k,v,o,gate,up,down_proj
-- of the Gemma language model only. SigLIP also has q/k/v_proj, and the
paper's adapter does not touch the vision tower; neither does this one.

The prompt is built exactly as vendor/VAMOS/server/vlm_server.py builds it
("<image><bos>" + prompt, through the same processor), so the adapter is
trained on the token sequence it will be asked at. The server loads any
model path with "lora" in it as an adapter on the base named inside it:

    bash start_server.sh --model_path <repo>/models/vamos_lora

Detours are rare: in the first collection 2.1% of frames leave the straight
line by more than 0.3 m, and those are the frames the fine-tune is for. Frames
that bulge more than DETOUR_M are repeated --oversample times in each epoch
(validation is left as collected, so its loss stays an honest average).

Training holds out 10% of the train *runs* (not frames: neighbouring frames
of one walk are nearly the same picture) for a validation loss, and keeps the
epoch that scored best on it. The Gate D routes are a separate split, never
seen here; scripts/vamos/evaluate.py is what scores them.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "vamos" / "train"
OUT = ROOT / "models" / "vamos_lora"
BASE = "mateoguaman/vamos"
DETOUR_M = 0.15      # m off the straight line that makes a frame worth repeating
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def device():
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")


def bulge(s):
    """How far the walked path strays from the straight line to its end,
    starting at the lens (scripts/vamos/evaluate.py measures the same)."""
    (ax, ay), (bx, by) = s["cam_pose"][:2], s["ground"][-1]
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy or 1e-12

    def off(px, py):
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
        return math.hypot(px - ax - t * dx, py - ay - t * dy)
    return max(off(px, py) for px, py in s["ground"])


def load_samples(data_dir):
    """Every labelled frame of every clean run, grouped by run."""
    index = json.loads((data_dir / "index.json").read_text())
    runs = {}
    for rid in index["clean_runs"]:
        lines = (data_dir / rid / "samples.jsonl").read_text().splitlines()
        runs[rid] = [dict(json.loads(l), image=str(data_dir / rid / json.loads(l)["frame"]))
                     for l in lines]
    return runs


def split(runs, val_frac, seed):
    """Whole runs to validation: a frame's neighbours are near-copies of it."""
    ids = sorted(runs)
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_frac))
    return ([s for r in ids[n_val:] for s in runs[r]],
            [s for r in ids[:n_val] for s in runs[r]], ids[:n_val])


def batch(processor, samples, dev):
    images = [Image.open(s["image"]).convert("RGB") for s in samples]
    enc = processor(text=["<image><bos>" + s["prompt"] for s in samples], images=images,
                    suffix=[s["target"] for s in samples], return_tensors="pt",
                    padding="longest")
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}


@torch.no_grad()
def evaluate(model, processor, samples, dev, bs):
    model.eval()
    total, n = 0.0, 0
    for k in range(0, len(samples), bs):
        chunk = samples[k:k + bs]
        total += model(**batch(processor, chunk, dev)).loss.item() * len(chunk)
        n += len(chunk)
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.1, help="fraction of runs held out")
    ap.add_argument("--limit", type=int, default=0, help="use only this many train frames")
    ap.add_argument("--oversample", type=int, default=4,
                    help=f"times each frame bulging over {DETOUR_M} m is seen per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if "lora" not in out.name.lower():
        # vlm_server.load_model only treats a path as an adapter if it says so.
        raise SystemExit(f"--out must have 'lora' in its name for the server to load it: {out}")

    torch.manual_seed(args.seed)
    dev = device()
    runs = load_samples(Path(args.data))
    train, val, val_runs = split(runs, args.val, args.seed)
    if args.limit:
        random.Random(args.seed).shuffle(train)
        train = train[:args.limit]
    detours = [s for s in train if bulge(s) > DETOUR_M]
    train = train + detours * (args.oversample - 1)
    print(f"{len(runs)} clean runs: {len(train)} train frames ({len(detours)} detours "
          f"x{args.oversample}), {len(val)} val frames ({len(val_runs)} runs held out), "
          f"device {dev}")

    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, target_modules=TARGETS, bias="none"))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = math.ceil(len(train) / (args.batch * args.accum)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(steps * 0.05, 1))
        * max(0.0, 1 - s / max(steps, 1)))

    best = evaluate(model, processor, val, dev, args.batch)
    print(f"epoch 0 (the base model): val loss {best:.4f}")
    history = [{"epoch": 0, "val_loss": best}]
    best_epoch, t0 = 0, time.time()
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        running, seen = 0.0, 0
        for k in range(0, len(train), args.batch):
            chunk = train[k:k + args.batch]
            loss = model(**batch(processor, chunk, dev)).loss
            (loss / args.accum).backward()
            running += loss.item() * len(chunk)
            seen += len(chunk)
            if (k // args.batch + 1) % args.accum == 0 or k + args.batch >= len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (k // args.batch) % 50 == 0:
                print(f"  epoch {epoch} {seen}/{len(train)} train loss {running / seen:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        v = evaluate(model, processor, val, dev, args.batch)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": v})
        print(f"epoch {epoch}: train loss {running / max(seen, 1):.4f}, val loss {v:.4f}")
        if v < best:
            best, best_epoch = v, epoch
            model.save_pretrained(out)
            print(f"  best so far -> {out}")

    if best_epoch == 0:
        print("no epoch beat the base model on validation loss; nothing saved")
        return
    data_hash = hashlib.sha256((Path(args.data) / "index.json").read_bytes()).hexdigest()
    (out / "training.json").write_text(json.dumps({
        "base": BASE, "data": args.data, "data_index_sha256": data_hash,
        "val_runs": val_runs, "train_frames": len(train), "val_frames": len(val),
        "best_epoch": best_epoch, "history": history,
        "args": vars(args), "device": dev, "wall_s": round(time.time() - t0),
    }, indent=2))
    print(f"kept epoch {best_epoch} (val loss {best:.4f}) -> {out}")


if __name__ == "__main__":
    main()
