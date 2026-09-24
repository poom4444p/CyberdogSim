"""Every filesystem location the stack uses, in one place.

Before this existed, four modules each derived the repo root for themselves
(`ROOT = dirname(dirname(abspath(__file__)))`) and then hardcoded the same
`map_tools/data/building` and `map_tools/config` strings into their own
constants. Moving a data directory meant finding all of them, and two of them
resolved the root to a different number of levels up, so they disagreed about
where the repo was whenever a file moved between directories.

Import from here instead. Paths are `pathlib.Path`, which `open()`, `os.path`
and PyYAML all accept directly; the handful of C extensions that insist on a
string (MuJoCo, imageio) get `str(...)` at the call site.

Generated output goes under `output/` and trained weights under `models/`,
both gitignored: nothing the repo tracks is ever written into the source tree.
"""
import os
from pathlib import Path

# src/cyberdog/paths.py -> src/cyberdog -> src -> the repo
ROOT = Path(__file__).resolve().parents[2]

# --- inputs the repo tracks ------------------------------------------------
CONFIG_DIR = ROOT / "config"
MAP_CONFIG = CONFIG_DIR / "map_config.yaml"
CAMERA_CONFIG = CONFIG_DIR / "camera_config.yaml"

DATA_DIR = ROOT / "data"
BUILDING_DIR = DATA_DIR / "building"
BUILDING_MAPS = BUILDING_DIR / "maps"
LOCATIONS_JSON = BUILDING_DIR / "locations.json"
ZONES_JSON = BUILDING_DIR / "zones.json"
SAMPLES_DIR = DATA_DIR / "samples"

DOCS_DIR = ROOT / "docs"
VENDOR_DIR = ROOT / "vendor"

# --- things that are generated, downloaded or trained ----------------------
MODELS_DIR = ROOT / "models"
LORA_ADAPTER = MODELS_DIR / "lora"                  # the gemma-2b-it fine-tune
LORA_OUTPUTS = MODELS_DIR / "lora_training_outputs"  # training checkpoints

DATASETS_DIR = ROOT / "datasets"
RAW_DATASET = DATASETS_DIR / "raw_dataset_english.json"
TRAINING_JSONL = DATASETS_DIR / "gemma_training_data.jsonl"

OUTPUT_DIR = ROOT / "output"
SCENE_CACHE = OUTPUT_DIR / "scene_cache"     # built MuJoCo XML and recorded MP4s
ROUTE_IMAGES = OUTPUT_DIR / "route_images"   # visualize_route.py drawings

BUILDING_SCENE = SCENE_CACHE / "building.xml"    # all three storeys and the lift
FLOOR1_SCENE = SCENE_CACHE / "floor1.xml"        # one storey, for run_demo

# --- the Go2 model, which lives outside the repo ---------------------------
# expanduser on the environment variable too, not just the default. The README
# sets it with `conda env config vars set MENAGERIE_PATH=~/...`, and conda
# stores that string literally -- no shell ever expands the tilde -- so without
# this the path is a directory called "~" and MuJoCo's error names a file that
# looks perfectly reasonable.
MENAGERIE = Path(os.path.expanduser(
    os.environ.get("MENAGERIE_PATH") or "~/mujoco_menagerie"))
GO2_XML = MENAGERIE / "unitree_go2" / "go2.xml"


def require_menagerie():
    """Check the Go2 model is where we think, and say so plainly if not.

    Worth a function because of *when* the failure otherwise lands. The scene
    builder writes this path into the XML as text and exits happily; the
    complaint arrives later, from MuJoCo, out of whichever run first loads the
    file -- naming a path nobody typed, on a line nobody wrote, in a file that
    was just reported as written successfully. Rebuilding the scene looks like
    the obvious fix and silently re-bakes the same wrong path.

    The usual cause is a shell that has not been re-activated since
    `conda env config vars set`, so the variable is configured and not yet in
    the environment. Checking here means the build refuses, and says which
    path it looked at and where that path came from.
    """
    if GO2_XML.exists():
        return
    src = ("MENAGERIE_PATH=" + os.environ["MENAGERIE_PATH"]
           if os.environ.get("MENAGERIE_PATH")
           else "the default, because MENAGERIE_PATH is not set in this shell")
    raise SystemExit(
        f"cannot find the Go2 model at {GO2_XML}\n"
        f"  looked there because of {src}\n"
        f"  fix: conda env config vars set MENAGERIE_PATH=/path/to/mujoco_menagerie "
        f"-n cyberdog_sim\n"
        f"       then `conda activate cyberdog_sim` again -- the variable only "
        f"reaches a shell on activation")

WALL_HEIGHT = 2.0       # metres, for the scene builder


def ensure_output():
    """Make the generated-output directories. Callers that write, call this."""
    for d in (SCENE_CACHE, ROUTE_IMAGES):
        d.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
