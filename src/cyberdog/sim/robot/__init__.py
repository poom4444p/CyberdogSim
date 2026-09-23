"""The Go2: pose, velocity commands, and the legs that fake them.

    robot_interface.py  the interface the real robot would also implement
    mujoco_robot.py     the twin's implementation -- kinematic body, real camera
    gait.py             a trot that makes the legs match the commanded velocity

The body is driven kinematically rather than by contact forces: honest for a
navigation PoC, and the reason dreaming.py's rollouts inherit its optimism.
"""
