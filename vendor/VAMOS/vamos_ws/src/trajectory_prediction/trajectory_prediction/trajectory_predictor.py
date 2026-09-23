import numpy as np
from dataclasses import dataclass
from typing import Optional
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path
import rospy
from trajectory_projection import Projector
from std_msgs.msg import Int32, Bool
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from visualization_msgs.msg import Marker
from enum import Enum

class NavState(Enum):
    IDLE                    = 0
    ROTATING_TO_VIEW_GOAL   = 1
    WAITING_FOR_CAMERA      = 2
    AWAITING_VLM_PREDICTION = 3
    EXECUTING_WAYPOINTS     = 4
    MISSION_COMPLETE        = 5

def state_to_marker(state: NavState, robot_frame: str) -> Marker:
    m = Marker()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = robot_frame
    m.ns = "robot_state"
    m.id = 0
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD

    m.pose.position.x = 0
    m.pose.position.y = 0
    m.pose.position.z = 1.0
    m.pose.orientation.w = 1.0
    m.scale.z = 0.4
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 1.0
    m.text = str(state.name)
    return m

@dataclass
class Packet:
    image: np.ndarray  # (height, width, channels)
    depth_image: np.ndarray  # (height, width)
    depth_intrinsics: np.ndarray  # (3, 3)
    camera_optical_to_vision_transform: TransformStamped # potentially stale transform from vision to camera optical frame
    goal_3d: np.ndarray  # (1, 3)
    goal_2d: Optional[np.ndarray] # (1, 2)  This gets populated in process_input
    timestamp: float
    value_map: np.ndarray  # (height, width)
    robot_pose_vision: Optional[PoseStamped] = None  # Robot pose in the vision frame (i.e. global frame)


class TrajectoryPredictorBase:
    def __init__(self, node_handle, tf_buffer: tf2_ros.Buffer, time_threshold=10000000):
        self.node_handle = node_handle
        self.current_packet = None
        self.last_packet = None # Used for timestamp of last VLM prediction
        self.projector = Projector()
        self.time_threshold = time_threshold # For timeout re-prediction

        self.current_nav_state = NavState.IDLE
        self.num_wps_to_complete = rospy.get_param("~num_waypoints_to_complete", 5)
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.global_frame = rospy.get_param("~global_frame", "vision")

        self.latest_reached_wp_idx = 0 
        self.completed_waypoints_count = 0
        self.current_vlm_output_2d: Optional[np.ndarray] = None # Stores 2D VLM output (K, 2)
        self.current_path_to_publish_3d: Optional[Path] = None # Stores the 3D Path msg to send

        self.side_offset_for_turn_cam = rospy.get_param("~side_offset_for_turn_cam", 0.5) # meters
        self.behind_offset_for_turn_cam = rospy.get_param("~behind_offset_for_turn_cam", 0.3) # meters

        self.post_rotation_delay = rospy.get_param("~post_rotation_delay", 2.0)
        self._camera_ready_time = None 
        self.path_seq_counter = 0
        
        self.path_topic = rospy.get_param("~path_topic", "/global_planner/planned_path")
        self.pose_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.state_marker_pub = rospy.Publisher("/robot_state_marker", Marker, queue_size=1)

        self.waypoint_reached_topic = rospy.get_param("~waypoint_reached_topic", "/carrot_picker/waypoint_reached_index")
        self.waypoint_reached_sub = rospy.Subscriber(
            self.waypoint_reached_topic, Int32, self.waypoint_reached_callback, queue_size=1
        )
        self.mission_complete_topic = rospy.get_param("~mission_complete_topic", "/nav_manager/mission_complete")
        self.mission_complete_sub = rospy.Subscriber(self.mission_complete_topic, Bool, self.mission_complete_callback, queue_size=1)
        self.mission_complete = False

        self.tf_buffer = tf_buffer

        rospy.loginfo(f"TrajectoryPredictorBase initialized. Num waypoints to complete: {self.num_wps_to_complete}, Time threshold: {self.time_threshold}s")
        rospy.loginfo("TrajectoryPredictorBase subscribed to /waypoint_reached_index")

    def waypoint_reached_callback(self, msg: Int32):
        """
        Callback for messages from /waypoint_reached_index.
        Updates the tracking of completed waypoints.
        """
        reached_idx = msg.data
        rospy.logdebug(f"Received waypoint reached feedback: index {reached_idx}")

        # Basic check: only update if the received index is what we expect or newer
        # and corresponds to the currently active trajectory.
        if self.current_path_to_publish_3d is not None and reached_idx > self.latest_reached_wp_idx:
            # We count waypoints as 0-indexed, so index `i` means `i+1` waypoints are done.
            # Example: if waypoint 0 is reached, 1 waypoint is complete.
            # If waypoint 1 is reached, 2 waypoints are complete.
            self.latest_reached_wp_idx = reached_idx
            self.completed_waypoints_count = self.latest_reached_wp_idx # + 1
            rospy.loginfo(f"\n\nUpdated completed_waypoints_count to: {self.completed_waypoints_count}\n\n")

    def mission_complete_callback(self, msg: Bool):
        """
        Callback for messages from /nav_manager/mission_complete.
        Updates the mission complete status.
        """
        if not self.mission_complete:
            self.mission_complete = msg.data
        if self.mission_complete:
            rospy.loginfo("Mission complete status received. Stopping trajectory prediction.")

    def _calculate_behind_and_side_wp_cam(self) -> Optional[np.ndarray]:
        """
        Calculates a single 3D waypoint in the camera_optical_frame,
        intended to make the robot turn. The waypoint is placed slightly
        behind the camera and offset to one side, opposite to the goal's
        x-position in the camera frame.

        Assumes self.current_packet and self.current_packet.goal_3d are valid.

        Returns:
            Optional[np.ndarray]: A (1, 3) numpy array for the waypoint, or None if
                                  essential data is missing.
        """
        if not (self.current_packet and self.current_packet.goal_3d is not None and \
                self.current_packet.goal_3d.shape == (1,3) ):
            rospy.logwarn_throttle(5.0, "TrajectoryPredictorBase: _calculate_behind_and_side_wp_cam missing or invalid current_packet.goal_3d.")
            return None

        goal_x_in_cam_optical = self.current_packet.goal_3d[0, 0]

        # If goal_x is positive (goal to camera's right), waypoint_x is negative (to camera's left), causing left turn.
        # If goal_x is negative (goal to camera's left), waypoint_x is positive (to camera's right), causing right turn.
        # If goal_x is zero, waypoint_x is zero (straight behind).
        turn_wp_x_cam = np.sign(goal_x_in_cam_optical) * self.side_offset_for_turn_cam
        turn_wp_y_cam = 0.0  # No vertical offset in camera frame for this waypoint
        turn_wp_z_cam = -self.behind_offset_for_turn_cam  # 'behind' is negative Z in typical camera optical frames

        return np.array([[turn_wp_x_cam, turn_wp_y_cam, turn_wp_z_cam]], dtype=np.float32)


    def process_input(self, current_packet: Packet = None):
        # ... (initial checks, goal projection, is_goal_in_image - same as previous Step 8)
        if current_packet is None: # Basic guard
            rospy.logwarn_throttle(5.0, "TPB.process_input called with None packet.")
            return
        self.current_packet = current_packet

        if self.projector.K is None:
            self.projector.set_intrinsics(self.current_packet.depth_intrinsics)
            self.projector.set_height_width(*self.current_packet.depth_image.shape)


        goal_2d_projected = self.projector.project_points_to_image(self.current_packet.goal_3d)
        is_goal_in_image = False
        if goal_2d_projected is not None and len(goal_2d_projected) > 0:
            self.current_packet.goal_2d = goal_2d_projected[0]
            is_goal_in_image = self.projector.is_point_in_image(goal_2d_projected)[0]
        else:
            rospy.logwarn_throttle(1.0, "TPB: Could not project 3D goal to 2D for visibility check.")
            self.current_packet.goal_2d = None

        next_nav_state = self.current_nav_state
        path_for_this_cycle: Optional[Path] = None

        state_marker = state_to_marker(self.current_nav_state, self.robot_frame)
        self.state_marker_pub.publish(state_marker)


        if self.current_nav_state == NavState.IDLE:
            rospy.loginfo_throttle(1.0, f"TPB State: {self.current_nav_state_to_str()}")
            self.completed_waypoints_count = 0
            self.latest_reached_wp_idx = 0
            self.current_vlm_output_2d = None
            self.current_path_to_publish_3d = None

            if self.mission_complete:
                rospy.loginfo("TPB: Mission complete. -> MISSION_COMPLETE")
                next_nav_state = NavState.MISSION_COMPLETE
                self.current_path_to_publish_3d = None
            elif not is_goal_in_image:
                rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Goal not in image. -> ROTATING_TO_VIEW_GOAL")
                next_nav_state = NavState.ROTATING_TO_VIEW_GOAL
            else:
                rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Goal in image. -> AWAITING_VLM_PREDICTION")
                next_nav_state = NavState.AWAITING_VLM_PREDICTION
        
        elif self.current_nav_state == NavState.ROTATING_TO_VIEW_GOAL:
            rospy.loginfo_throttle(1.0, f"TPB State: {self.current_nav_state_to_str()}")
            if is_goal_in_image:
                rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Goal now in view. -> WAITING_FOR_CAMERA for {self.post_rotation_delay}s")
                # schedule the earliest time we’ll trust the VLM
                self._camera_ready_time = rospy.Time.now() + rospy.Duration(self.post_rotation_delay)
                next_nav_state = NavState.WAITING_FOR_CAMERA
                self.current_path_to_publish_3d = None
                
                empty_path = Path() 
                empty_path.header.stamp = rospy.Time.now()
                if self.current_packet and self.current_packet.camera_optical_to_vision_transform:
                     empty_path.header.frame_id = self.current_packet.camera_optical_to_vision_transform.header.frame_id
                else: # Fallback if packet/transform is somehow unavailable here
                     empty_path.header.frame_id = self.global_frame
                empty_path.header.seq = self.path_seq_counter
                self.path_seq_counter += 1
                path_for_this_cycle = empty_path
            else:
                # Goal is still Out Of View (OOV). Predict a turn trajectory.
                rospy.loginfo_throttle(1.0, f"TPB: {self.current_nav_state_to_str()}: Goal still OOV. Predicting turn trajectory.")
                
                # This method now directly returns 3D waypoints in global frame.
                traj_3d_vision_rotation = self.predict_turn_to_goal_trajectory()

                if traj_3d_vision_rotation is not None and traj_3d_vision_rotation.shape[0] > 0:
                    current_path_id = self.path_seq_counter
                    self.path_seq_counter += 1

                    path_msg = self._convert_vision_points_to_path(
                        traj_3d_vision_rotation,
                        current_path_id,
                        rospy.Time.now() # Path timestamp
                    )
                    
                    if path_msg:
                        self.current_path_to_publish_3d = path_msg 
                        path_for_this_cycle = path_msg
                        rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Generated turn path with {len(path_msg.poses)} waypoints.")
                    else:
                        rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Failed to convert 3D vision turn points to Path message.")
                        self.current_path_to_publish_3d = None 
                else:
                    rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: predict_turn_to_goal_trajectory returned None or empty.")
                    self.current_path_to_publish_3d = None
        
        elif self.current_nav_state == NavState.WAITING_FOR_CAMERA:
            rospy.loginfo_throttle(1.0, "TPB: Waiting for camera to stabilize")
            if rospy.Time.now() >= self._camera_ready_time:
                rospy.loginfo("TPB: Camera stable; now going to VLM.")
                next_nav_state = NavState.AWAITING_VLM_PREDICTION
            else:
                # keep sending the empty path (or nothing) until ready
                return
            
        elif self.current_nav_state == NavState.AWAITING_VLM_PREDICTION:
            rospy.loginfo(f"TPB State: {self.current_nav_state_to_str()}. Predicting trajectory.")
            # Reset waypoint tracking for this new prediction
            self.completed_waypoints_count = 0
            self.latest_reached_wp_idx = 0
            self.current_vlm_output_2d = None 

            predicted_3d_waypoints_vision = self.predict_trajectory() # Now returns 3D points in global frame

            if predicted_3d_waypoints_vision is not None and predicted_3d_waypoints_vision.shape[0] > 0:
                # Points are already in global frame. Create Path message.
                current_path_id = self.path_seq_counter
                self.path_seq_counter += 1
                
                path_msg_global = self._convert_global_points_to_path(
                    predicted_3d_waypoints_vision,
                    current_path_id,
                    rospy.Time.now() # Path message timestamp
                )

                if path_msg_global:
                    self.current_path_to_publish_3d = path_msg_global
                    path_for_this_cycle = self.current_path_to_publish_3d # Publish this new path
                    self.last_packet = self.current_packet # Store packet used for this prediction
                    rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: New trajectory predicted with {len(path_msg_global.poses)} waypoints. -> EXECUTING_WAYPOINTS")
                    next_nav_state = NavState.EXECUTING_WAYPOINTS
                else:
                    rospy.logwarn(f"TPB: {self.current_nav_state_to_str()}: Failed to convert predicted 3D global points to Path message.")
                    next_nav_state = NavState.IDLE
                    self.current_path_to_publish_3d = None
                    # self.current_vlm_output_2d already None or cleared
            else:
                rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: predict_trajectory (VLM) returned None or empty 3D waypoints. -> IDLE")
                next_nav_state = NavState.IDLE # Prediction failed, go back to IDLE
                self.current_path_to_publish_3d = None
                # self.current_vlm_output_2d already None or cleared

        elif self.current_nav_state == NavState.EXECUTING_WAYPOINTS:
            rospy.loginfo_throttle(1.0, f"TPB State: {self.current_nav_state_to_str()}. Comp WPs: {self.completed_waypoints_count}/{self.num_wps_to_complete}")

            if self.current_path_to_publish_3d is None or not self.current_path_to_publish_3d.poses or self.last_packet is None:
                rospy.logwarn(f"TPB: {self.current_nav_state_to_str()}: Missing critical data (path or last_packet). -> IDLE")
                next_nav_state = NavState.IDLE
                self.current_path_to_publish_3d = None
            else:
                # Condition 1: Waypoints completed for the current segment/plan
                # Use the number of poses in the currently executing path.
                num_total_wps_in_current_path = len(self.current_path_to_publish_3d.poses)
                
                # Re-plan if N waypoints are done OR if all predicted waypoints are done
                waypoints_segment_done = (self.completed_waypoints_count >= self.num_wps_to_complete) or \
                                         (self.completed_waypoints_count >= num_total_wps_in_current_path and num_total_wps_in_current_path > 0)


                # Condition 2: Timeout since last VLM prediction
                time_since_last_prediction = self.current_packet.timestamp - self.last_packet.timestamp
                timeout_reached = time_since_last_prediction >= self.time_threshold

                if waypoints_segment_done:
                    rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Waypoint segment completed ({self.completed_waypoints_count} waypoints). -> IDLE")
                    next_nav_state = NavState.IDLE
                elif timeout_reached:
                    rospy.loginfo(f"TPB: {self.current_nav_state_to_str()}: Timeout reached ({time_since_last_prediction:.2f}s). -> IDLE")
                    next_nav_state = NavState.IDLE

                if waypoints_segment_done or timeout_reached:
                    self.current_path_to_publish_3d = None
                
                    empty_path = Path() 
                    empty_path.header.stamp = rospy.Time.now()
                    if self.current_packet and self.current_packet.camera_optical_to_vision_transform:
                        empty_path.header.frame_id = self.current_packet.camera_optical_to_vision_transform.header.frame_id
                    else:
                        empty_path.header.frame_id = self.global_frame
                    empty_path.header.seq = self.path_seq_counter
                    self.path_seq_counter += 1
                    path_for_this_cycle = empty_path
                

        elif self.current_nav_state == NavState.MISSION_COMPLETE:
            rospy.loginfo_throttle(1.0, f"TPB State: {self.current_nav_state_to_str()}. Mission complete.")
            next_nav_state = NavState.MISSION_COMPLETE
        
        else:
            rospy.logerr(f"TPB: Unknown navigation state: {self.current_nav_state_to_str()}. Resetting to IDLE.")
            next_nav_state = NavState.IDLE

        self.current_nav_state = next_nav_state

        # ... (Unified Path Publishing - same as previous Step 8, uses path_for_this_cycle.header.seq)
        if path_for_this_cycle is not None:
            self.pose_pub.publish(path_for_this_cycle)
            rospy.loginfo(f"TPB: Published path (ID {path_for_this_cycle.header.seq}) for state {self.current_nav_state_to_str()} with {len(path_for_this_cycle.poses)} poses.")

    
    def _convert_global_points_to_path(
        self,
        points_3d_global: np.ndarray,
        path_id: int,
        path_stamp: rospy.Time # Timestamp for the Path header and individual poses
    ) -> Optional[Path]:
        """
        Converts a numpy array of 3D points in the global frame into a
        nav_msgs/Path message.

        Args:
            points_3d_global: (N, 3) numpy array of points in global frame.
            path_id: Sequence ID for the Path message.
            path_stamp: Timestamp for the Path message header and its poses.

        Returns:
            A nav_msgs/Path message, or None if input points are invalid.
        """
        if points_3d_global is None or points_3d_global.shape[0] == 0:
            rospy.logwarn("TrajectoryPredictorBase: _convert_global_points_to_path received no points.")
            return None
        
        if not (isinstance(points_3d_global, np.ndarray) and \
                points_3d_global.ndim == 2 and points_3d_global.shape[1] == 3):
            rospy.logwarn(
                f"TrajectoryPredictorBase: _convert_global_points_to_path input 'points_3d_global' "
                f"is not a valid (N,3) numpy array. Shape: {points_3d_global.shape if isinstance(points_3d_global, np.ndarray) else type(points_3d_global)}"
            )
            return None

        path_msg = Path()
        path_msg.header.stamp = path_stamp
        path_msg.header.frame_id = self.global_frame
        path_msg.header.seq = path_id

        for point_global in points_3d_global:
            pose_global = PoseStamped()
            pose_global.header.stamp = path_stamp
            pose_global.header.frame_id = self.global_frame
            
            pose_global.pose.position.x = float(point_global[0])
            pose_global.pose.position.y = float(point_global[1])
            pose_global.pose.position.z = float(point_global[2])
            pose_global.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose_global)
            
        return path_msg

    def predict_trajectory(self) -> Optional[np.ndarray]:
        """
        Predicts a 3D trajectory for forward navigation, returning points
        in the global frame.
        
        To be implemented by subclasses.
        
        Returns:
            Optional[np.ndarray]: A (K, 3) numpy array of 3D waypoints in the
                                  global frame, or None on failure or if no
                                  trajectory can be determined.
        """
        raise NotImplementedError(
            "predict_trajectory must be implemented by subclasses and return "
            "a (K,3) numpy array in global frame or None."
        )

    def predict_turn_to_goal_trajectory(self) -> Optional[np.ndarray]:
        """
        Predicts a 3D trajectory, typically a single waypoint, for turning
        the robot, returning points in the global frame.
        
        To be implemented by subclasses.
        
        Returns:
            Optional[np.ndarray]: An (M, 3) numpy array of 3D waypoints in the
                                  global frame (often M=1 for a single turn
                                  waypoint), or None on failure or if no
                                  trajectory can be determined.
        """
        raise NotImplementedError(
            "predict_turn_to_goal_trajectory must be implemented by subclasses and return "
            "an (M,3) numpy array in global frame or None."
        )

    def current_nav_state_to_str(self) -> str:
        """Helper to convert current_nav_state enum to string for logging."""
        return self.current_nav_state.name

class DummyTrajectoryPredictor(TrajectoryPredictorBase):
    def __init__(self, node_handle, tf_buffer: tf2_ros.Buffer, time_threshold=1000000):
        super().__init__(node_handle, tf_buffer, time_threshold)
    

    def predict_trajectory(self) -> Optional[np.ndarray]:
        if not (self.current_packet and \
                self.current_packet.goal_2d is not None and \
                self.projector.K is not None and \
                self.projector.img_height > 0 and self.projector.img_width > 0 and \
                self.current_packet.depth_image is not None and \
                self.current_packet.camera_optical_to_vision_transform is not None):
            rospy.loginfo_throttle(5.0, "DummyTrajectoryPredictor: predict_trajectory called with missing essential data or uninitialized projector.")
            return None

        # 1. Generate 2D dummy points
        goal_2d = self.current_packet.goal_2d
        
        if self.projector.img_width == 0 or self.projector.img_height == 0:
             rospy.loginfo("DummyTrajectoryPredictor: Projector image dimensions are zero.")
             return None

        start_x = self.projector.img_width / 2.0
        start_y = self.projector.img_height - 1.0 # Start at the bottom center

        goal_x = goal_2d[0]
        goal_y = goal_2d[1]

        num_points = 5
        path_points_2d_list = []

        t_values = np.linspace(0, 1, num_points + 1)
        t_values = t_values[1:] # Skip t=0 (start point itself)
        for t in t_values:
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)
            path_points_2d_list.append([x, y])
        
        if not path_points_2d_list:
            rospy.loginfo("DummyTrajectoryPredictor: No 2D points generated.")
            return None
        
        traj_2d_np = np.array(path_points_2d_list, dtype=np.float32)

        # 2. Project to 3D in camera_optical_frame
        traj_3d_cam_optical = self.projector.get_3d_points(
            traj_2d_np, self.current_packet.depth_image
        )

        if traj_3d_cam_optical is None or traj_3d_cam_optical.shape[0] == 0:
            rospy.loginfo("DummyTrajectoryPredictor: Failed to project 2D dummy points to 3D in camera_optical_frame.")
            return None

        # 3. Transform to 3D in global frame
        cam_to_vision_transform = self.current_packet.camera_optical_to_vision_transform
        points_3d_vision_list = []

        for point_cam_opt in traj_3d_cam_optical:
            pose_cam_optical = PoseStamped()
            pose_cam_optical.header.stamp = cam_to_vision_transform.header.stamp # Use transform's stamp for consistency
            pose_cam_optical.header.frame_id = cam_to_vision_transform.child_frame_id # Source frame
            
            pose_cam_optical.pose.position.x = float(point_cam_opt[0])
            pose_cam_optical.pose.position.y = float(point_cam_opt[1])
            pose_cam_optical.pose.position.z = float(point_cam_opt[2])
            pose_cam_optical.pose.orientation.w = 1.0 # Default orientation

            try:
                # do_transform_pose can raise tf2_ros.TransformException if the transform is invalid,
                # but here we assume the provided transform from the packet is usable.
                # Other errors could be due to invalid PoseStamped.
                transformed_pose_vision = do_transform_pose(pose_cam_optical, cam_to_vision_transform)
                p = transformed_pose_vision.pose.position
                points_3d_vision_list.append([p.x, p.y, p.z])
            except Exception as e: # Catching a broad exception if do_transform_pose fails
                rospy.loginfo(
                    f"DummyTrajectoryPredictor: Failed to transform point from "
                    f"'{cam_to_vision_transform.child_frame_id}' to '{cam_to_vision_transform.header.frame_id}'. "
                    f"Point: {point_cam_opt}. Error: {e}"
                )
                return None # If one point fails, the whole trajectory transformation fails

        if not points_3d_vision_list:
            rospy.loginfo("DummyTrajectoryPredictor: No points successfully transformed to global frame.")
            return None
            
        return np.array(points_3d_vision_list, dtype=np.float32)


    def predict_turn_to_goal_trajectory(self) -> Optional[np.ndarray]:
        """
        Calculates a 3D waypoint in the global frame intended to make the
        robot turn towards the goal.
        It uses a 'behind-and-side' waypoint calculated in camera_optical_frame
        and then transforms it to the global frame.
        """
        rospy.loginfo("DummyTrajectoryPredictor: Predicting turn-to-goal trajectory.")

        if not (self.current_packet and \
                self.current_packet.camera_optical_to_vision_transform is not None):
            rospy.loginfo_throttle(5.0, "DummyTrajectoryPredictor: predict_turn_to_goal_trajectory missing essential packet data for transformation.")
            return None

        # 1. Calculate waypoint in camera_optical_frame using the base class method
        # super() ensures we call TrajectoryPredictorBase._calculate_behind_and_side_wp_cam()
        wp_3d_cam_optical_array = super()._calculate_behind_and_side_wp_cam()

        if wp_3d_cam_optical_array is None or wp_3d_cam_optical_array.shape[0] == 0:
            rospy.loginfo("DummyTrajectoryPredictor: _calculate_behind_and_side_wp_cam returned None or empty for turning.")
            return None
        
        # wp_3d_cam_optical_array is expected to be (1,3)
        point_cam_opt = wp_3d_cam_optical_array[0] 

        # 2. Transform this waypoint to the global frame
        cam_to_vision_transform = self.current_packet.camera_optical_to_vision_transform
        
        pose_cam_optical = PoseStamped()
        pose_cam_optical.header.stamp = cam_to_vision_transform.header.stamp
        pose_cam_optical.header.frame_id = cam_to_vision_transform.child_frame_id # Source frame for transformation
        
        pose_cam_optical.pose.position.x = float(point_cam_opt[0])
        pose_cam_optical.pose.position.y = float(point_cam_opt[1])
        pose_cam_optical.pose.position.z = float(point_cam_opt[2])
        pose_cam_optical.pose.orientation.w = 1.0 # Default orientation

        try:
            transformed_pose_vision = do_transform_pose(pose_cam_optical, cam_to_vision_transform)
            p = transformed_pose_vision.pose.position
            # Return as a (1, 3) numpy array, consistent with other trajectory outputs
            return np.array([[p.x, p.y, p.z]], dtype=np.float32)
        except Exception as e: # Catching a broad exception if do_transform_pose fails
            rospy.loginfo(
                f"DummyTrajectoryPredictor: Failed to transform 'behind-and-side' point from "
                f"'{cam_to_vision_transform.child_frame_id}' to '{cam_to_vision_transform.header.frame_id}'. "
                f"Point: {point_cam_opt}. Error: {e}"
            )
            return None