"""Building the MuJoCo scene the dog walks around in.

    levels.py       storey heights and slab thickness -- one source of truth
    build_scene.py  grids + meshes -> the MuJoCo XML, and camera params
    lift.py         the lift car and shaft, so floors connect
    stairs.py       the stairwells (which the router refuses to route down)
    obstacles.py    crates deliberately ABSENT from the map, so the
                    perception layer has something real to find
    pedestrians.py  ...and people, absent from it as well, who also move,
                    which is a different problem and a different answer

The last two are the experiment: everything else in the scene is also in the
occupancy grid, which would make perception decorative. A crate is gone round.
A person is waited for.
"""
