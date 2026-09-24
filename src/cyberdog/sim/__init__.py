"""The digital twin: MuJoCo, a virtual Mid-360, and VAMOS in the loop.

Shared pieces, which the entry points and each other all read:

    control.py     the control law and its gains -- also what dreaming.py
                   imagines, so the dream and the dog agree
    clearance.py   static clearance from the map (walls + no-go zones)
    overlay.py     drawing onto the camera frame
    vamos_client.py  the VLM as a service: 5 candidate paths -> 1 chosen

Subpackages:

    scene/    building the MuJoCo XML: storeys, lift, stairs, obstacles
    robot/    the Go2 itself -- kinematics, gait, pose interface
    sensing/  what the dog perceives: lidar -> perception/tracking -> dreaming

Entry points:

    run_building.py   the full stack, three storeys, lift, obstacle avoidance
    run_demo.py       one floor, no video -- the shortest closed loop
    record_demo.py    one floor, with an MP4
"""
