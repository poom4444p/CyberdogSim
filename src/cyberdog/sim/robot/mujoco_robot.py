"""MuJoCo stand-in for the Cyberdog.

The real dog walks itself -- we send body velocities and its own controller
handles the legs. So this drives the body kinematically instead of simulating
a gait: same interface, none of the wasted work.
"""
import math
import mujoco
import numpy as np

from cyberdog.sim.robot import gait
from cyberdog.sim.scene.build_scene import camera_params
from cyberdog.sim.robot.robot_interface import RobotInterface


class MujocoRobot(RobotInterface):
    CAM_W       = 640
    CAM_H       = 480
    CAM_FORWARD = 0.34         # lens sits on the head, ahead of body centre

    CONTROL_DT  = 0.05         # 20 Hz, a realistic cmd_vel rate
    MAX_V       = 0.8          # m/s    -- the real dog's limits live here
    MAX_W       = 1.5          # rad/s

    def __init__(self, scene_xml, start_xy=(45.0, 4.0), start_yaw=math.pi / 2,
                 start_z=0.0):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self._vx = self._vy = self._w = 0.0
        self.sim_time = 0.0
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "dogcam")
        self._renderer = None   # built on first get_image, it is not cheap
        # Height comes from the projector's config so the two can't disagree
        # about where the lens is.
        _, self.CAM_HEIGHT = camera_params()
        # Standing height of the "home" stance. In a stacked building qpos[2]
        # is that plus whichever floor the dog is on, so keep the two apart.
        self.BASE_Z = float(self.model.key_qpos[0][2])
        self.z = float(start_z)
        # Gait state. The legs are posed from how far the body has actually
        # travelled, so this is the only thing that has to be carried between
        # ticks; see gait.py for why it is drawn rather than simulated.
        self._phase, self._amp = 0.0, 0.0
        self.reset(start_xy, start_yaw)

    def reset(self, xy, yaw):
        # The "home" keyframe folds the legs into a standing stance; without it
        # the dog spawns with its legs straight out.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0] = xy[0]
        self.data.qpos[1] = xy[1]
        self.data.qpos[2] = self.BASE_Z + self.z
        # qpos[3:7] is the orientation quaternion. Yaw only, so x and y stay 0.
        self.data.qpos[3] = math.cos(yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = math.sin(yaw / 2)
        self._place_camera(xy[0], xy[1], yaw)
        mujoco.mj_forward(self.model, self.data)

    def set_height(self, z):
        """Height of the surface under the dog -- floor level, or partway up a
        flight of stairs. Everything else is still planned in 2D."""
        self.z = float(z)

    def get_pose(self):
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = math.atan2(2 * (qw * qz + qx * qy),
                         1 - 2 * (qy * qy + qz * qz))
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(yaw)

    def set_velocity(self, vx, vy=0.0, w=0.0):
        self._vx = float(np.clip(vx, -self.MAX_V, self.MAX_V))
        self._vy = float(np.clip(vy, -self.MAX_V, self.MAX_V))
        self._w = float(np.clip(w, -self.MAX_W, self.MAX_W))

    def step(self):
        x, y, yaw = self.get_pose()
        dt = self.CONTROL_DT

        # vx/vy are body-frame (like cmd_vel), qpos is world-frame, so rotate.
        # Yaw first, so the dog moves along the heading it ends the tick with.
        yaw += self._w * dt
        x += (self._vx * math.cos(yaw) - self._vy * math.sin(yaw)) * dt
        y += (self._vx * math.sin(yaw) + self._vy * math.cos(yaw)) * dt

        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = self.BASE_Z + self.z
        self.data.qpos[3] = math.cos(yaw / 2)
        self.data.qpos[6] = math.sin(yaw / 2)
        self._walk(math.hypot(self._vx, self._vy) * dt, self._w * dt)
        self._place_camera(x, y, yaw)
        mujoco.mj_forward(self.model, self.data)
        self.sim_time += dt

    def place(self, xy, yaw, z=None):
        """Put the dog exactly here and count it as one control tick.

        The lift is scripted, not driven: cmd_vel has no z, and the real dog's
        own controller handles boarding. So those legs set the pose directly
        and everything else keeps going through step(). The gait is driven off
        the movement either way, so the dog still walks in and out of the car.
        """
        x0, y0, yaw0 = self.get_pose()
        if z is not None:
            self.z = float(z)
        self._walk(math.dist((x0, y0), xy), self._wrap(yaw - yaw0))
        self.data.qpos[0] = float(xy[0])
        self.data.qpos[1] = float(xy[1])
        self.data.qpos[2] = self.BASE_Z + self.z
        self.data.qpos[3] = math.cos(yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = math.sin(yaw / 2)
        self._place_camera(self.data.qpos[0], self.data.qpos[1], yaw)
        mujoco.mj_forward(self.model, self.data)
        self.sim_time += self.CONTROL_DT

    def _walk(self, ds, dyaw):
        """Pose the legs for a body that just moved `ds` metres and turned `dyaw`.

        Amplitude follows the movement with a lag rather than switching on and
        off, so the dog settles into the home stance when it stops instead of
        freezing mid-stride, and picks the stride back up smoothly.
        """
        moving = 1.0 if (abs(ds) + abs(dyaw) * gait.TURN_R) > 1e-4 else 0.0
        self._amp += (moving - self._amp) * 0.2
        self._phase = gait.advance(self._phase, ds, dyaw)
        self.data.qpos[7:19] = gait.pose(self._phase, self._amp)

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def move_mocap(self, name, pos):
        """Put a mocap body somewhere -- the lift car, which rides with the dog.

        Mocap bodies sit outside qpos, so adding one to the scene leaves the
        Go2's keyframe and this class's qpos indices alone. No-op if the scene
        has no such body, which is every single-floor scene.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return
        self.data.mocap_pos[self.model.body_mocapid[bid]] = pos
        mujoco.mj_forward(self.model, self.data)

    def _place_camera(self, x, y, yaw):
        """Point the onboard camera along the body heading.

        Must run before mj_forward: model.cam_* is the definition, data.cam_x*
        is what the renderer reads, and only mj_forward copies one to the other.
        Get it backwards and the first frame renders from (0,0,0) -- solid black,
        which looks exactly like a lighting bug.
        """
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        zax = -fwd                             # MuJoCo cameras look down -z
        yax = np.array([0.0, 0.0, 1.0])        # up
        xax = np.cross(yax, zax)               # right
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.column_stack([xax, yax, zax]).flatten())
        # Callers project from camera_pose(), which applies the same offset,
        # so the pixel the projector computes matches what this camera sees.
        self.model.cam_pos[self._cam_id] = [x + self.CAM_FORWARD * fwd[0],
                                            y + self.CAM_FORWARD * fwd[1],
                                            self.z + self.CAM_HEIGHT]
        self.model.cam_quat[self._cam_id] = q

    def camera_pose(self):
        """Pose of the lens, not the body.

        The projector works from wherever the camera actually is, so anything
        that projects a goal pixel must use this and not get_pose().
        """
        x, y, yaw = self.get_pose()
        return (x + self.CAM_FORWARD * math.cos(yaw),
                y + self.CAM_FORWARD * math.sin(yaw), yaw)

    def get_image(self):
        """Onboard RGB frame - what VAMOS will be looking at."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.CAM_H, self.CAM_W)
        self._renderer.update_scene(self.data, camera="dogcam")
        return self._renderer.render()
