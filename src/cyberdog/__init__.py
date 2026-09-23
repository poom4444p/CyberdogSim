"""Cyberdog blind-navigation PoC: a robotic guide dog, in a digital twin.

Four layers, in the order a spoken command travels through them:

    language/   "take me to room 201" -> structured destinations
                (gemma-2b-it + LoRA, plus deterministic splitting/floor rules)
    mapping/    occupancy grids, semantic Behavior Layer zones, A*
    planning/   multi-floor routing to checkpoints, and projecting those
                checkpoints into camera pixels for the VLM
    sim/        the MuJoCo twin: robot, LiDAR, perception, VAMOS in the loop

Design principle, inherited from the spec (docs/spec.md): *AI proposes, simple
code disposes.* Open-world understanding lives in the AI modules; every
safety-critical decision is deterministic.

See README.md for how to run each layer and docs/architecture.md for how the
data actually flows between them.
"""
__version__ = "0.1.0"
