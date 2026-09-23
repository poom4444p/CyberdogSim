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
MENAGERIE = Path(os.environ.get("MENAGERIE_PATH",
                                os.path.expanduser("~/mujoco_menagerie")))
GO2_XML = MENAGERIE / "unitree_go2" / "go2.xml"

WALL_HEIGHT = 2.0       # metres, for the scene builder


def ensure_output():
    """Make the generated-output directories. Callers that write, call this."""
    for d in (SCENE_CACHE, ROUTE_IMAGES):
        d.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
