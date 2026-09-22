import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENAGERIE = os.environ.get("MENAGERIE_PATH", os.path.expanduser("~/mujoco_menagerie"))
GO2_XML = os.path.join(MENAGERIE, "unitree_go2", "go2.xml")

BUILDING = os.path.join(ROOT, "map_tools", "data", "building", "maps")
WALL_HEIGHT = 2.0