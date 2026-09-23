"""The line between navigation logic and whatever is actually moving.

The planner, the projector and VAMOS only ever call these four methods, so
swapping the simulator for the real Cyberdog is one new subclass and nothing
else changes.
"""


class RobotInterface:
    def get_image(self):
        """Camera frame, HxWx3 uint8 RGB."""
        raise NotImplementedError

    def get_pose(self):
        """(x, y, yaw) in the map frame -- metres and radians."""
        raise NotImplementedError

    def set_velocity(self, vx, vy, w):
        """Body-frame command: forward m/s, sideways m/s, yaw rad/s."""
        raise NotImplementedError

    def step(self):
        """Advance one control tick."""
        raise NotImplementedError


class RealCyberdog(RobotInterface):
    """Hardware-day checklist -- nothing here runs yet.

    get_image     <- /camera/color/image_raw
    get_pose      <- /odom or FAST-LIO pose, quaternion -> yaw
    set_velocity  -> geometry_msgs/Twist on /cmd_vel
    step          -> rospy.Rate(20).sleep()
    """
