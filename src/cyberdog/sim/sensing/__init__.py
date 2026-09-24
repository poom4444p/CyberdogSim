"""What the dog can actually perceive -- and what it does about it.

The three read in order, and the order is the point:

    lidar.py       a virtual Mid-360. Raycasts the scene. Knows nothing about
                   meaning -- it reports surfaces.
    perception.py  throws away every return the static map already explains,
                   and offers the rest as clearance. Also moves the goal when
                   the goal is inside a crate (`free_destination`), which is what
                   actually makes the dog go round.
    tracking.py    clusters those returns and matches them to last tick's, so
                   that a thing has a velocity. Without it a walking person
                   and a post are the same reading, and they want opposite
                   answers: go round the post, wait for the person.
    dreaming.py    rolls the real control law forward with noise to score a
                   candidate path: P(no collision) x room left.

Sensor, interpretation, prediction -- kept separate so that the day a real
Mid-360 arrives, only lidar.py changes.
"""
