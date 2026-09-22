# Input Treating Layer

The NLU front-end of the CyberDog pipeline. Turns a natural-language command
into structured intent JSON that the Planner Layer (`main_planner.py`, one
level up) consumes:

```json
{"target_location": "room 310", "task": "navigation", "query": null}
```

It's a LoRA fine-tune of `unsloth/gemma-2b-it` trained on a synthetic,
template-generated dataset (no real user recordings involved), with two
small **rule-based pre-processors** in front of it for things the model
wasn't trained on: multiple destinations in one command, and floor phrases.

```
"I need to pee upstairs, then the library and the caf"
        │
        ▼ command_splitter.py      (rules)
["I need to pee upstairs", "Go to the library", "Go to the caf"]
        │   for each stop:
        ▼ floor_parser.py          (rules)
floor=2, "I need to pee"
        │
        ▼ infer.parse_command      (model)
{"target_location": "restroom", "task": "navigation", "query": null}
        │
        ▼ planner_layer/ (+ floor=2)
```

## Pipeline

Each script is a step; run them in this order to go from scratch to a working model.

| Step | Script | Input → Output |
|---|---|---|
| 1 | `generate_dataset.py` | (templates) → `raw_dataset_english.json` |
| 2 | `prepare_dataset.py` | `raw_dataset_english.json` → `gemma_training_data.jsonl` |
| 3 | `train_local_mac.py` | `gemma_training_data.jsonl` → `mac_lora_final/` (+ checkpoints in `mac_lora_outputs/`) |
| 4 | `test_model.py` | interactive manual sanity check |
| 5 | `eval_model.py` | batch accuracy check against N fresh generated samples |

`infer.py` isn't a pipeline step — it's the shared inference module
(`load_model`, `parse_command`, `generate_raw`, `get_device`) that
`main_planner.py`, `test_model.py`, `eval_model.py` and
`planner_layer/test_locations.py` all import so the model-loading/generation
logic lives in exactly one place.

Other files:

| File | What it is |
|---|---|
| `command_splitter.py` | Rules: split a multi-stop command into single-stop commands (no model) |
| `floor_parser.py` | Rules: pull out a floor phrase and turn it into a floor number (no model) |
| `archive/raw_dataset_english_old_1000.json` | Old 1,000-sample dataset, kept for reference; nothing loads it |
| `mac_lora_outputs/` | Intermediate training checkpoints (only needed to resume a run) |

## What it recognizes

Locations fall into three groups, sampled roughly evenly so the numerous
room numbers don't drown out everything else:

- **Numbered classrooms** — `room 101`...`room 310`, generated from
  `NUM_FLOORS = 3` / `ROOMS_PER_FLOOR = 10` in `generate_dataset.py`
  (floor 3, room 10 → `"310"`). **Placeholder** — edit those two constants
  to match the real building.
- **Department labs** — `computer engineering lab`, `electrical engineering lab`,
  `mechanical engineering lab`, `chemistry lab`, `biology lab`, `physics lab`.
  **Placeholder names** — swap in the university's real department names
  in `lab_locations` / `LAB_ALIASES` once confirmed.
- **Other** — `hallway`, `library`, `office`, `main entrance`, `cafeteria`,
  `restroom`, `server room`.

The alias tables `OTHER_ALIASES` / `LAB_ALIASES` live at the top of
`generate_dataset.py` (module level) because `command_splitter.py` imports
them to recognize place names. Add an alias there and both the training data
and the splitter pick it up. The building map
(`map_tools/scripts/generate_building.py`) also uses these exact canonical
names — `planner_layer/test_locations.py` checks they match.

Every location also has **casual aliases** it must resolve back to the
canonical name (e.g. `"lib"` → `library`, `"caf"` → `cafeteria`, `"310"` in
context → `room 310`). A small set of **implicit slang** phrases imply a
destination without naming any room at all (`"I need to pee"` → `restroom`,
`"I'm starving"` → `cafeteria`, `"let's hit the books"` → `library`) — see
`implicit_slang` in `generate_dataset.py` to add more.

Two task types: `navigation` (go somewhere, `query` is `null`) and
`visual_qa` (go somewhere and answer a yes/no question about it, `query`
holds that question).

## Running it

Everything here needs `torch` + `transformers` + `peft` + `trl` + `datasets`,
which are not in the base Python — use a conda/venv environment that has
them (this project used a `vamos_mac` conda env during development).

```bash
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS-only libomp workaround, harmless elsewhere
cd input_treating_layer

python3 generate_dataset.py        # -> raw_dataset_english.json
python3 prepare_dataset.py         # -> gemma_training_data.jsonl
python3 train_local_mac.py         # -> mac_lora_final/ (overwrites it)
python3 test_model.py              # interactive: type a command, see the raw output
python3 eval_model.py --num-samples 100   # batch accuracy report
```

### Linux / NVIDIA GPU users

Setup (once):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft trl datasets accelerate
# NVIDIA GPU: install the CUDA build of torch from pytorch.org first
python3 -c "import torch; print(torch.cuda.is_available())"   # should print True
```

Run every script from inside `input_treating_layer/` — the data and model
paths are relative. Each file also has a "Running on Linux" note at the top.
Older NVIDIA GPUs without bfloat16 (pre-Ampere, e.g. GTX 10xx / T4): change
`torch_dtype=torch.bfloat16` to `torch.float16` in `infer.py` and
`train_local_mac.py`.

Otherwise nothing needs to change. `infer.get_device()` checks `cuda` → `mps` → `cpu`
in that order, so a Linux box with an NVIDIA GPU automatically trains and
runs on `cuda`. The two Mac-specific bits already in the code are both
no-ops elsewhere:
- `KMP_DUPLICATE_LIB_OK=TRUE` works around a macOS libomp conflict.
- `train_local_mac.py` is named `_mac` for history's sake, not because it's
  Mac-only.

## Rule-based pre-processing

The model only knows one destination per command and has never seen a floor
phrase ("I want to pee in the second floor" made it output
`"second floor"` as the location). Rather than retrain for these while the
building layout is still a placeholder, two rule-based steps run first. Both
are plain Python with no dependencies, and both have a self-test.

### `command_splitter.py` — multiple destinations

`split_destinations(text) -> list[str]`

| Split on | Example |
|---|---|
| `then`, `and then`, `after that`, `afterwards`, `followed by`, `next`, `;` | "the library, then the cafeteria" |
| `and` + movement verb | "... and go to the cafeteria" |
| `and` / `,` / `as well as` / `plus` + a place | "the library and the caf", "the lib, the caf, and room 310", "room 105 and 310" |

A segment without a verb gets `Go to` added ("the cafeteria" → "Go to the
cafeteria"), `both` is dropped, `rooms 105` becomes `room 105`, and a bare
`310` becomes `room 310`. A "place" is any alias from `OTHER_ALIASES` /
`LAB_ALIASES` or a room number.

**Not** split: `and check if ...` (a question about the same stop), `next to`
(describes a place), and commas in the training templates ("Check the
library, the door is open").

### `floor_parser.py` — floors

`extract_floor(text, current_floor=None) -> (floor or None, cleaned_text)`

| Kind | Phrases |
|---|---|
| Absolute | "second floor", "2nd floor", "floor 2", "floor two", "level 2", "2F", "ground floor" (= 1), "top floor" (= `NUM_FLOORS`) |
| Relative | "upstairs", "downstairs", "one floor up", "two floors down", "up a floor", "the floor above/below" |

The phrase is removed before the text goes to the model
("Take me to the 2nd floor library" → `2`, "Take me to the library").
Relative phrases need `current_floor` — the robot's floor from localization
(in the demo, the floor the previous stop ended on); without it they raise
`ValueError`. The result isn't range-checked here: "upstairs" on the top
floor returns `NUM_FLOORS + 1` and the planner reports it.

### Safety check

Neither rule step may change a command the model was trained on. Verified on
12,000 generated training commands: 0 split, 0 floor matches. Re-check after
editing the rules:

```bash
python3 -c "
import random; from generate_dataset import generate_synthetic_dataset as g
from command_splitter import split_destinations as sp; from floor_parser import extract_floor as fl
random.seed(1); s=g(4000)
print(sum(len(sp(x['text']))!=1 or fl(x['text'],1)[0] is not None for x in s), 'changed')"
```

### Self-tests

```bash
cd input_treating_layer
python3 command_splitter.py   # 24 cases
python3 floor_parser.py       # 22 cases
```

## Training config (`train_local_mac.py`)

- Base model: `unsloth/gemma-2b-it`, bfloat16.
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`, targets
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- 3 epochs, batch size 1 × grad accumulation 4, learning rate 2e-4.
- A **fixed-size** eval split (up to 150 samples, not a %) so eval cost
  doesn't balloon as the dataset grows. Evaluates every 100 steps.
- `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` —
  the checkpoint actually saved to `mac_lora_final/` is whichever step had
  the lowest eval loss, not necessarily the last one. This is the main
  guard against overfitting; watch `eval_loss` in the training output and
  make sure it's still trending down (if it turns around and climbs while
  train loss keeps dropping, that's overfitting).

## Latest results

Batch eval (`eval_model.py --num-samples 100`) against the current
`mac_lora_final/`, on freshly generated samples covering all 43 locations
(rooms, labs, other, slang):

| Metric | Score |
|---|---|
| Valid JSON output | 100/100 |
| `target_location` correct | 100/100 |
| `task` correct | 100/100 |
| `query` correct | 100/100 |
| **Fully correct (all 3 fields)** | **100/100** |

Caveat: `eval_model.py` draws samples from the *same templates* used to
build the training set, so this measures in-distribution correctness, not
generalization to phrasing nobody wrote a template for. An earlier,
smaller-dataset version of this model (9 generic locations, no slang) broke
on question-form commands like `"Is the door open in the lab? Go check."`
that weren't well represented in the templates — worth spot-checking new
phrasing styles with `test_model.py` before trusting this on truly novel
input.

## Known limitations / next steps

- Floors and multiple stops are handled by rules, not the model. Phrasings
  outside those rules aren't understood: "one level up from here", "the
  place where you eat, then ...", slang joined by "and" ("I need to pee and
  eat"), or a question covering several places ("check if the lights are on
  in the library and the cafeteria" only asks about the library). A command
  that's *only* a direction ("go upstairs") leaves no destination for the
  model. Once the real building is known, retrain with a `floor` field and
  multi-stop examples (the rules can generate that training data).
- Department lab names and the floor/room-count layout are placeholders —
  swap in the real ones.
- `office` and `server room` have no slang aliases (couldn't think of
  realistic informal phrasing for them — add some if you have real
  examples).
- The dataset is entirely synthetic/template-generated, not real recorded
  or crowd-sourced commands — accuracy here doesn't guarantee accuracy on
  how actual users phrase things.
- `mac_lora_outputs/` holds intermediate training checkpoints (useful only
  to resume a specific run); safe to delete except the most recent one if
  disk space matters. `mac_lora_final/` is the only one anything else
  in the repo actually loads.
