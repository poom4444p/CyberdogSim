"""A trot, drawn rather than simulated.

The dog is driven kinematically -- body velocities in, pose out, no contacts
(see mujoco_robot.py). That is the right call for a navigation twin, but it
left the legs frozen in the "home" stance, sliding along the floor like
furniture. Anyone watching the video had to take it on trust that this was a
robot that walks.

So the legs are posed from the distance the body has actually covered. It is
not locomotion -- nothing here holds the dog up, and on the real Go2 the
onboard controller owns the gait entirely. It is an honest drawing of one:
step where the body moves, stand still where it does not.

Trot, because that is what a Go2 does at these speeds: diagonal pairs swing
together, FL with RR and FR with RL.

Geometry. Each leg is two 0.213 m links in the sagittal plane, and the joint
angles come from placing the foot and solving for the knee. Working forwards
from the home stance (thigh 0.9, calf -1.8) puts the foot 0.265 m directly
below the hip, which is what the keyframe means by standing.
"""
import math

L1 = L2 = 0.213                 # thigh and calf, metres, from the Go2 model
STAND_H = (L1 + L2) * math.cos(0.9)   # foot below hip in the home stance

STRIDE = 0.34                   # metres of ground covered per full cycle
LIFT = 0.055                    # how high a swinging foot clears the floor
TURN_R = 0.28                   # a pivot of 1 rad counts as this much stride

# Model joint order is FL, FR, RL, RR; a trot pairs the diagonals.
LEGS = ("FL", "FR", "RL", "RR")
PHASE = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}


def advance(phase, ds, dyaw):
    """Next gait phase after the body moved `ds` metres and turned `dyaw`.

    Turning on the spot covers no ground but is still stepping, so it feeds
    the same phase -- otherwise the dog pirouettes with its feet planted.
    """
    travel = abs(ds) + abs(dyaw) * TURN_R
    return (phase + 2 * math.pi * travel / STRIDE) % (2 * math.pi)


def leg_angles(phase, amplitude=1.0):
    """(hip, thigh, calf) for one leg at this phase of the cycle.

    The foot traces a flattened oval: back along the ground through the
    stance half, forward through the air in the swing half. `amplitude` fades
    the whole thing out, so a standing dog settles into the home stance
    instead of freezing mid-step.
    """
    swing = max(0.0, math.sin(phase))
    x = -amplitude * (STRIDE / 2) * math.cos(phase)
    z = -STAND_H + amplitude * LIFT * swing
    return (0.0,) + _ik(x, z)


def pose(phase, amplitude=1.0):
    """All twelve joint angles, in the model's FL, FR, RL, RR order."""
    out = []
    for leg in LEGS:
        out += list(leg_angles((phase + PHASE[leg]) % (2 * math.pi), amplitude))
    return out


def _ik(x, z):
    """Foot at (x, z) relative to the hip -> (thigh, calf) angles.

    Two-link inverse kinematics, knee bending backwards, which is the only
    one of the two solutions a dog's hind leg can reach.
    """
    r = min(math.hypot(x, z), L1 + L2 - 1e-6)
    cos_knee = (L1 * L1 + L2 * L2 - r * r) / (2 * L1 * L2)
    calf = -(math.pi - math.acos(max(-1.0, min(1.0, cos_knee))))
    beta = math.atan2(L2 * math.sin(calf), L1 + L2 * math.cos(calf))
    return math.atan2(-x, -z) - beta, calf
