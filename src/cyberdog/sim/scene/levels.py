"""How tall the stack is -- one set of numbers for every floor.

stairs.py, lift.py and build_scene.py all need to agree on where floor n's
walking surface sits, so the heights live here and nowhere else.
"""

FLOOR_HEIGHT = 2.2              # slab top to slab top; walls are 2.0 m
SLAB_THICK = 0.12               # each upper floor's plate, hanging below its level


def floor_z(n):
    """World height of floor n's walking surface."""
    return (n - 1) * FLOOR_HEIGHT
