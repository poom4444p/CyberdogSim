"""The line between navigation logic and whatever is actually moving.

The planner, the projector, perception and VAMOS only ever call the methods
below, so swapping the simulator for the real Cyberdog is one new subclass and
nothing else changes.

Not on this line, and sim-only: place(), set_height(), move_mocap(s)(),
has_body() and model/data. They script what the real dog does by itself (the
lift ride, the stairs) or what the real world does (the people, the lift car),
and the ground-truth checks the robot is never shown. On hardware there is
nothing to call in their place.
"""


class RobotInterface:
    def get_image(self):
        """Camera frame, HxWx3 uint8 RGB."""
        raise NotImplementedError

    def get_pose(self):
        """(x, y, yaw) in the map frame -- metres and radians."""
        raise NotImplementedError

    def camera_pose(self):
        """(x, y, yaw) of the lens, not the body, in the map frame."""
        raise NotImplementedError

    def get_scan(self):
        """LiDAR returns this tick: (N, 3) points in the map frame."""
        raise NotImplementedError

    def now(self):
        """Seconds since the run started, on the clock the control loop runs on."""
        raise NotImplementedError

    def set_velocity(self, vx, vy, w):
        """Body-frame command: forward m/s, sideways m/s, yaw rad/s."""
        raise NotImplementedError

    def step(self):
        """Advance one control tick."""
        raise NotImplementedError


# The twin is mujoco_robot.MujocoRobot. The real dog will be a subclass that
# talks to the Go2 board (go2_board, on the J4012) for velocity and pose.
