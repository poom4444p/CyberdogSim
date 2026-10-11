# VAMOS LoRA fine-tune (spec L4 step 5)

Zero-shot VAMOS spreads its five candidates about ±0.25 m over a 2 m path, so
any sidestep wider than that is still the map's to make (`sim/README.md`, *Who
actually does the avoiding*). This folder is the fine-tune that is meant to
move it to the model.

| Step | Command | Env | Time |
|---|---|---|---|
| 1. Collect train data | `python scripts/vamos/collect.py` | `cyberdog_sim` | ~11 min, 4 jobs (216 runs) |
| 2. Collect held-out data | `python scripts/vamos/collect.py --test` | `cyberdog_sim` | ~1 min |
| 3. Train | `python scripts/vamos/train_lora.py` | `vamos_mac` (or a CUDA box) | hours on a GPU; far longer on a Mac |
| 4. Score base, then adapter | `python scripts/vamos/evaluate.py --name base` then `--name lora --load <repo>/models/vamos_lora` | `cyberdog_sim` + server | ~10 min each |
| 5. Closed loop | in `vendor/VAMOS/server`: `bash start_server.sh --model_path <repo>/models/vamos_lora` (absolute path), then from the repo root `python scripts/gate_d.py` | both | a Gate D campaign |

The server keeps the last model it loaded, including one loaded by
`evaluate.py --load`. So step 4's `--name base` must come before any `--load`,
or carry `--load mateoguaman/vamos`. Restart the server before each step 5
campaign. `gate_d.py` asks the server's `/health` which model it serves,
prints it, keeps it as `model` in `summary.json`, and warns if it changed
before the campaign ended.

**Labels.** `sim/autolabel.py` (`run_building --autolabel DIR`) keeps a frame
every 0.5 s of a map-only run, and labels it with the path the dog *walked*
from there: 5 points by arc length, projected into that frame, starting where
the camera can first see the floor and cut where the path leaves the image.
Each frame keeps two prompts: `prompt`, whose goal is that path's end
(hindsight), and `map_prompt`, the projector's goal, which is what the live
stack sends. Only clean runs are used: arrived, with no collision, no handle
contact and no person contact.

**Prompt.** `train_lora.py` and `evaluate.py` use `map_prompt` by default
(`--prompt hindsight` for the other). The hindsight prompt hands the model
the answer's end: the first adapter, trained on it, learnt to draw a straight
line there. It scored 0.26 → 0.03 top-path error with that prompt, no better
than base with the live one, and failed Gate D
(`docs/dreaming_safety_kpis.md`, 2026-10-11). Results in
`output/vamos_eval/` from before this change were scored with `prompt` unless
the name ends in `_map`.

**Split.** The 7 Gate D routes are held out entirely (`--test`). They share
corridors with the train routes, so this is a held-out route, not a held-out
layout. Within the train set, `train_lora.py` holds out 10% of *runs* for its
validation loss.

**Adapter.** r=16, α=16, dropout 0.05 on q/k/v/o/gate/up/down_proj of the
Gemma language model; SigLIP is untouched. The prompt is built through the
same processor call as `vlm_server.py`, so training sees the exact tokens
inference does. The output directory must have `lora` in its name, since that
is how the server decides to load a path as an adapter.

What the labels teach is the map-only stack's own walk, which is clean on
every Gate D route but not perfect. A model trained on it can at best match
it. The aim is for VAMOS to propose the detours itself, rather than only
surviving the gate.
