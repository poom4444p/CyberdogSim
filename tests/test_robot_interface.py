"""Tests for sim/robot: the twin keeps to the line the real dog will plug into."""
import math
import os

import numpy as np
import pytest

from cyberdog import paths
from cyberdog.sim.robot.robot_interface import RobotInterface

SCENE = str(paths.BUILDING_SCENE)
needs_scene = pytest.mark.skipif(not os.path.exists(SCENE),
                                 reason="no building scene: build_scene --building")

# Everything run_building's brain reads from the robot. A real robot that
# implements these needs nothing else from the twin.
BRAIN = ["get_image", "get_pose", "camera_pose", "get_scan", "now",
         "set_velocity", "step"]


def test_the_interface_names_everything_the_brain_reads():
    for name in BRAIN:
        assert callable(getattr(RobotInterface, name)), name


@pytest.fixture(scope="module")
def robot():
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot
    from cyberdog.sim.scene import levels
    return MujocoRobot(SCENE, start_xy=(21.0, 9.8), start_yaw=math.pi,
                       start_z=levels.floor_z(2))


@needs_scene
class TestMujocoRobot:
    def test_implements_every_brain_method(self, robot):
        for name in BRAIN:
            assert getattr(type(robot), name) is not getattr(RobotInterface, name), name

    def test_scan_is_the_lidar_from_where_the_dog_stands(self, robot):
        # What run_building used to do itself, from the pose it had just read.
        from cyberdog.sim.sensing.lidar import Lidar
        x, y, yaw = robot.get_pose()
        expected = Lidar(robot.model, robot.data).scan((x, y), robot.z, yaw)
        got = robot.get_scan()
        assert got.shape == expected.shape and len(got) > 0
        np.testing.assert_array_equal(got, expected)

    def test_now_is_sim_time_and_steps_with_the_tick(self, robot):
        t0 = robot.now()
        robot.set_velocity(0.0, 0.0, 0.0)
        robot.step()
        assert robot.now() == pytest.approx(t0 + robot.CONTROL_DT)
        assert robot.now() == robot.sim_time
