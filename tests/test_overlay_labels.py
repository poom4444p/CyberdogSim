"""Tests for the chase-panel room labels in overlay.py.

These are a caption on the video, so what is worth pinning is that the
projection agrees with the camera that rendered the frame, and that two names
never interleave into a third unreadable one.

No mujoco here: `ChaseCam` reads plain floats off whatever it is handed, which
is deliberate -- `overlay.py` is the one module in `sim/` that stays free of
it, so the same drawing code works on a frame from anywhere.
"""
import numpy as np
import pytest

from cyberdog.sim.overlay import LABEL_H, ChaseCam, label_places


class FakeEye:
    """One eye of MuJoCo's stereo pair, with the fields ChaseCam reads."""

    def __init__(self, pos, forward, up, near=0.16, top=0.0659):
        self.pos, self.forward, self.up = pos, forward, up
        self.frustum_near, self.frustum_top = near, top


class FakeScene:
    """`MjvScene.camera` is a two-entry array: left eye, right eye."""

    def __init__(self, pos_l, pos_r, forward, up):
        self.camera = [FakeEye(pos_l, forward, up),
                       FakeEye(pos_r, forward, up)]


def looking_east(dist=3.0):
    """A camera `dist` west of (20, 9.5, 0.5), looking along +x.

    The eyes straddle the axis in y, the way MuJoCo offsets them.
    """
    return FakeScene([20.0 - dist, 9.534, 0.5], [20.0 - dist, 9.466, 0.5],
                     [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])


class TestChaseCam:
    def test_lookat_lands_dead_centre(self):
        """The point the camera is aimed at is the middle of the frame."""
        cam = ChaseCam(looking_east(), 640, 480)
        assert cam.pixel((20.0, 9.5, 0.5)) == (320, 240)

    def test_stereo_midpoint_not_one_eye(self):
        """Taking a single eye puts everything a few pixels off centre.

        Small enough to read as a broken projection rather than the wrong
        camera, which is exactly why it is worth a test.
        """
        scene = looking_east()
        cam = ChaseCam(scene, 640, 480)
        one_eye = np.array(scene.camera[0].pos)
        assert not np.allclose(cam.pos, one_eye)
        assert np.allclose(cam.pos, [17.0, 9.5, 0.5])

    def test_behind_the_camera_is_none(self):
        """A room already walked past does not project onto the frame."""
        cam = ChaseCam(looking_east(), 640, 480)
        assert cam.pixel((5.0, 9.5, 0.5)) is None

    def test_right_of_centre_is_right_of_centre(self):
        """Looking along +x, smaller y is to the camera's right."""
        cam = ChaseCam(looking_east(), 640, 480)
        u, _ = cam.pixel((20.0, 8.5, 0.5))
        assert u > 320

    def test_higher_is_higher(self):
        """Image y grows downwards, so a point above lookat has a smaller v."""
        cam = ChaseCam(looking_east(), 640, 480)
        _, v = cam.pixel((20.0, 9.5, 2.0))
        assert v < 240


class TestLabelPlaces:
    def blank(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draws_a_nearby_room(self):
        img = self.blank()
        cam = ChaseCam(looking_east(), 640, 480)
        label_places(img, cam, [("library", (22.0, 9.5))], (20.0, 9.5), 0.0)
        assert img.any(), "a room 2 m ahead should be named"

    def test_far_rooms_are_not_drawn(self):
        """The corridor is 48 m long; naming all of it is a wall of text."""
        img = self.blank()
        cam = ChaseCam(looking_east(), 640, 480)
        label_places(img, cam, [("library", (44.0, 9.5))], (20.0, 9.5), 0.0)
        assert not img.any()

    def test_overlapping_labels_do_not_interleave(self):
        """Two doors a few centimetres apart: one name, not two mangled ones.

        The regression this guards is "main entrance" and "library" drawn on
        top of each other coming out as "mai library nce".
        """
        cam = ChaseCam(looking_east(), 640, 480)
        pair = [("main entrance", (24.0, 9.50)), ("library", (24.2, 9.52))]

        both = self.blank()
        label_places(both, cam, pair, (20.0, 9.5), 0.0)

        nearer = self.blank()
        label_places(nearer, cam, pair[:1], (20.0, 9.5), 0.0)

        assert np.array_equal(both, nearer), \
            "the farther of two colliding names should be dropped, not drawn"

    def test_nearer_name_is_the_one_kept(self):
        img = self.blank()
        cam = ChaseCam(looking_east(), 640, 480)
        far_first = [("library", (24.2, 9.52)), ("main entrance", (24.0, 9.50))]
        label_places(img, cam, far_first, (20.0, 9.5), 0.0)

        only_near = self.blank()
        label_places(only_near, cam, [("main entrance", (24.0, 9.50))],
                     (20.0, 9.5), 0.0)
        assert np.array_equal(img, only_near)

    def test_labels_sit_above_the_floor(self):
        """LABEL_H puts a name over the door, not down where the dog is."""
        cam = ChaseCam(looking_east(), 640, 480)
        door = (23.0, 9.5)
        at_head = cam.pixel((door[0], door[1], LABEL_H))
        at_foot = cam.pixel((door[0], door[1], 0.0))
        assert at_head[1] < at_foot[1]

    def test_nothing_to_say_leaves_the_frame_alone(self):
        img = self.blank()
        cam = ChaseCam(looking_east(), 640, 480)
        out = label_places(img, cam, [], (20.0, 9.5), 0.0)
        assert not out.any()


class TestDoorSigns:
    """`Run.doors` turning a routing table into a signboard.

    locations.json exists to answer "where do I walk to", so two things in it
    are fine for routing and wrong for signage: facing rooms share a door on
    the centreline, and one place carries several names.
    """

    def signs(self, floor):
        from cyberdog.planning.building_router import BuildingRouter
        from cyberdog.sim.run_building import Run

        class Bare(Run):            # no scene, no robot -- doors() needs neither
            def __init__(self, router):
                self.router, self._doors = router, {}

        return dict(Bare(BuildingRouter()).doors(floor))

    def test_facing_rooms_get_their_own_side(self):
        """room 101 and room 106 share door_xy [3.0, 9.5] exactly.

        Signed on the centreline they are one box, so the overlap rule drops
        one and half the corridor goes unlabelled.
        """
        s = self.signs(1)
        assert "room 101" in s and "room 106" in s
        assert s["room 101"][1] != s["room 106"][1]
        assert s["room 101"][0] == s["room 106"][0], "same door, opposite walls"
        # One each side of the 9.5 m centreline.
        assert (s["room 101"][1] - 9.5) * (s["room 106"][1] - 9.5) < 0

    def test_signs_land_inside_the_corridor(self):
        """On the wall, not through it: the walls are at y 8.15 and 10.85."""
        for xy in self.signs(1).values():
            assert 8.15 <= xy[1] <= 10.85

    def test_synonyms_collapse_to_one_sign(self):
        """stairs / staircase / stairway / stairwell are one stairwell."""
        s = self.signs(1)
        assert "stairs" in s
        for alias in ("staircase", "stairway", "stairwell"):
            assert alias not in s, f"{alias} is a second sign on one door"

    def test_hallway_is_not_a_room(self):
        assert "hallway" not in self.signs(1)

    def test_every_floor_is_signed(self):
        for floor in (1, 2, 3):
            assert self.signs(floor), f"floor {floor} has no signs"
