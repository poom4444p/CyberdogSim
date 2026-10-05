"""The affordance module (spec Layer 5): can the dog walk *there*, physically?

The clearance field in sim/clearance.py answers that question from the map.
This package is where the answer from the terrain itself will come from: an
MLP trained on what a real locomotion policy actually managed to walk over
in Isaac Lab.

    data.py    the contract between the two halves. The elevation patch
               (grid, units, which way up), the trial record, and the rule
               that turns a record into a label. Pure numpy, no Isaac and no
               torch, so the Isaac collector and the runtime on the Mac build
               the patch with the same code.

The collector itself lives in scripts/isaac/, because it only runs inside
Isaac Lab on an NVIDIA GPU.
"""
