#!/usr/bin/env python3
'''
This node is in charge of taking in a 3D goal PoseStamped published by nav_manager.py in the /spot/goal topic
and it outputs a velocity for the low-level control policy to track as a Twist topic published to /spot/cmd_vel.

This node interfaces between the long-range goal or set of waypoints, publishes the necessary data for a neural network model to predict a path in image space, and manages this set of waypoints from which waypoints are selected sequentially, and either switches to the next waypoint once the waypoint is reached if it hasn't obtained a new path message from the neural network, or resets the waypoints to the new ones predicted by the neural network.    

In more detail, it takes in the 3D goal PoseStamped, and it first tries to project it into the rgb image that the robot will use for inference. If the projected point does not lie on the image, it simply publishes a twist to cmd_vel that rotates the robot toward the goal until the goal is in sight. Once the projected goal is in the 2d image, it publishes the image, and the 2d coordinates for the goal, and saves the associated depth image and the position of the robot in vision frame at that timestep. A separate node (vlm_inference.py) subscribes to the rgb image and the 2d goal coordinates and publishes a 2d path message containing 2d image coordinates for the path to follow. This node then takes in those predictions, converts them to 3D waypoints in the optical camera frame using the trajectory_projection package in misc, then shifts them if needed to account for robot motion while the neural network was running inference to convert them into waypoints in the base frame of the robot, and manages the waypoints: track the closest waypoint ahead of the robot, and if the robot is close enough to the waypoint, switch to the next waypoint. If the robot is not close enough to the waypoint, it will keep trying to reach that waypoint until it gets a new path message from the neural network, at which point it will reset the waypoints to the new ones predicted by the neural network. To convert the current waypoint to a velocity, it uses a simple P controller to track the target point, and publishes the velocity to cmd_vel.
'''

import rospy
import numpy as np
import cv2
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from trajectory_projection import Projector

class VLMNavigationNode:
    def __init__(self):
        # Parameters from navigate.yaml
        self.image_topic = rospy.get_param("~image_topic", "/zed2/zed_node/left/image_rect_color")
        self.image_camera_info_topic = rospy.get_param("~image_camera_info", "/zed2/zed_node/left/camera_info")
        self.depth_topic = rospy.get_param("~depth_topic", "/zed2/zed_node/depth/depth_registered")
        self.depth_camera_info_topic = rospy.get_param("~depth_camera_info", "/zed2/zed_node/depth/camera_info")
        self.goal_topic = rospy.get_param("~goal_topic", "/spot/goal")
        self.odom_topic = rospy.get_param("~odom_topic", "/spot/odometry_corrected")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/spot/cmd_vel")
        self.goal_2d_topic = rospy.get_param("~goal_2d", "/vamos/goal_2d")
        self.vlm_input_image_topic = rospy.get_param("~vlm_input_image", "/vamos/input_image")
        
        # Additional parameters
        self.path_2d_topic = rospy.get_param("~vlm_path_2d_topic", "/vamos/path_2d")
        self.waypoint_distance_threshold = rospy.get_param("~waypoint_distance_threshold", 0.5)
        self.rotation_speed = rospy.get_param("~rotation_speed", 0.5)
        self.visualization_enabled = rospy.get_param("~visualization_enabled", True)
        
        # Control parameters (from CarrotPickerNode)
        self.carrot_radius = rospy.get_param('~carrot_radius', 0.1)
        self.max_x_vel = rospy.get_param('~max_x_vel', 1.0)
        self.max_y_vel = rospy.get_param('~max_y_vel', 1.0)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.0)
        self.p_lin_x = rospy.get_param('~p_lin_x', 1.0)
        self.p_lin_y = rospy.get_param('~p_lin_y', 1.0)
        self.p_ang = rospy.get_param('~p_angular', 1.0)
        self.fwd_angle_thresh = rospy.get_param('~fwd_angle_thresh', 90)
        
        # Variables
        self.cv_bridge = CvBridge()
        self.camera_info = None
        self.depth_camera_info = None
        self.projector = None
        self.current_goal = None
        self.current_image = None
        self.current_depth = None
        self.current_odom = None
        self.waypoints_3d = []
        self.current_waypoint_index = 0
        self.cmd_vel = None
        self.goal_in_sight = False
        
        # Time stamps for synchronized data
        self.image_timestamp = None
        self.depth_timestamp = None
        self.robot_pose_at_capture = None
        self.capture_transform = None
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        rospy.Subscriber(self.image_camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(self.depth_camera_info_topic, CameraInfo, self.depth_camera_info_callback)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber(self.path_2d_topic, Float32MultiArray, self.path_2d_callback)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.goal_2d_pub = rospy.Publisher(self.goal_2d_topic, Float32MultiArray, queue_size=10)
        self.vlm_input_image_pub = rospy.Publisher(self.vlm_input_image_topic, Image, queue_size=10)
        
        # Visualization publishers
        if self.visualization_enabled:
            self.waypoints_pub = rospy.Publisher("/visualization/waypoints", MarkerArray, queue_size=10)
            self.current_waypoint_pub = rospy.Publisher("/visualization/current_waypoint", Marker, queue_size=10)
            self.goal_marker_pub = rospy.Publisher("/visualization/goal", Marker, queue_size=10)
        
        # TODO: Clean up
        # vlm_predictot = VLMTrajectoryPredictor
        # message_filter sub
        # end TODO

        rospy.loginfo("VLM Navigation Node Initialized")

    # def message_filter_callback(self, blah1, blah2, ...):
    #     image = cv_bridge()

    #     packet = Packet(
    #         image,
    #         etc
    #     )
    #     vlm_predictor.current_packet = packet
    #     vlm_predictor.process_input()

    def camera_info_callback(self, msg):
        self.camera_info = msg
        if self.camera_info and not self.projector:
            K = self.camera_info.K
            self.projector = Projector(K)
            rospy.loginfo("Camera projector initialized")

    def depth_camera_info_callback(self, msg):
        self.depth_camera_info = msg
    
    def image_callback(self, msg):
        self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_timestamp = msg.header.stamp

        # Store the transform at the time of image capture for later use
        ## TODO: Fix this. Need transform from base_frame to vision frame. Unnecessary if 32d path is published in vision frame. Inference node needs to receive the transform at which the image was taken in vision frame.  
        if self.current_odom:
            try:
                self.capture_transform = self.tf_buffer.lookup_transform(
                    "base_link",
                    self.camera_info.header.frame_id if self.camera_info else msg.header.frame_id,
                    msg.header.stamp,
                    rospy.Duration(0.1)
                )
                self.robot_pose_at_capture = self.current_odom.pose.pose
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Could not store camera transform: {e}")
        
        self.process_goal_projection()
    
    def depth_callback(self, msg):
        self.current_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_timestamp = msg.header.stamp
    
    def goal_callback(self, msg):
        self.current_goal = msg
        self.goal_in_sight = False
        self.waypoints_3d = []  # Clear previous waypoints  # NOTE: Do we need to clear waypoints everytime the goal changes? VLM might not be fast enough to handle this 
        self.current_waypoint_index = 0
        
        # Publish visualization if enabled
        if self.visualization_enabled:
            self.publish_goal_visualization()
            
        self.process_goal_projection()
    
    def odom_callback(self, msg):
        self.current_odom = msg
        self.process_waypoint_tracking()
    
    def path_2d_callback(self, msg):
        """Handle path received from VLM inference node in 2D image coordinates"""
        if self.current_depth is None or self.capture_transform is None:
            rospy.logwarn("Missing required data to process 2D path")
            return
        
        # Extract 2D points from Float32MultiArray
        points_2d = np.array(msg.data).reshape(-1, 2)
        
        if len(points_2d) == 0:
            rospy.logwarn("Received empty path, ignoring")
            return
            
        rospy.loginfo(f"Received 2D path with {len(points_2d)} points")
        
        # Convert 2D image points to 3D points in camera frame using depth
        points_3d = self.projector.get_3d_points(points_2d, self.current_depth)
        
        if len(points_3d) == 0:
            rospy.logwarn("No valid 3D points could be extracted from path")
            return
            
        # Create waypoints from the 3D points
        self.create_waypoints_from_3d_points(points_3d)
        
        # Reset waypoint tracking
        self.current_waypoint_index = 0
        rospy.loginfo(f"Created path with {len(self.waypoints_3d)} valid waypoints")
        
        # Publish visualization if enabled
        if self.visualization_enabled:
            self.publish_waypoint_visualization()
    
    def create_waypoints_from_3d_points(self, points_3d):
        """Transform 3D points from camera frame to base frame and create waypoints"""
        try:
            self.waypoints_3d = []
            
            for point in points_3d:
                # Create a PoseStamped for the point
                pose = PoseStamped()
                pose.header.frame_id = self.camera_info.header.frame_id
                pose.header.stamp = self.image_timestamp
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.position.z = point[2]
                
                # Default orientation
                pose.pose.orientation.w = 1.0
                
                # Transform pose using stored transform from image capture time
                transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, self.capture_transform)
                self.waypoints_3d.append(transformed_pose)
            
            # Calculate orientations to face the next waypoint
            self.calculate_waypoint_orientations()
                
        except Exception as e:
            rospy.logwarn(f"Error creating waypoints: {e}")
    
    def calculate_waypoint_orientations(self):
        """Calculate orientations for each waypoint to face the next point"""
        if len(self.waypoints_3d) <= 1:
            return
            
        # For each waypoint (except the last), set orientation to face the next waypoint
        for i in range(len(self.waypoints_3d) - 1):
            current = self.waypoints_3d[i].pose.position
            next_point = self.waypoints_3d[i+1].pose.position
            
            # Calculate yaw angle to face next waypoint
            dx = next_point.x - current.x
            dy = next_point.y - current.y
            yaw = np.arctan2(dy, dx)
            
            # Convert to quaternion
            q = quaternion_from_euler(0, 0, yaw)
            self.waypoints_3d[i].pose.orientation.x = q[0]
            self.waypoints_3d[i].pose.orientation.y = q[1]
            self.waypoints_3d[i].pose.orientation.z = q[2]
            self.waypoints_3d[i].pose.orientation.w = q[3]
        
        # Set the last waypoint's orientation to match the goal if available
        if self.current_goal:
            self.waypoints_3d[-1].pose.orientation = self.current_goal.pose.orientation
    
    def process_goal_projection(self):
        """Project 3D goal to 2D image coordinates and handle goal visibility"""
        if self.current_goal is None or self.current_image is None or self.projector is None:
            return
            
        try:
            # Try to project 3D goal to 2D image coordinates
            goal_in_camera = self.transform_goal_to_camera_frame()
            if goal_in_camera is None:
                self.handle_goal_out_of_sight()
                return
                
            # Check if the projected point is within image bounds
            h, w = self.current_image.shape[:2]
            x, y = self.project_point_to_image(goal_in_camera)
            
            if 0 <= x < w and 0 <= y < h:
                self.goal_in_sight = True
                self.publish_goal_2d(x, y)
                self.publish_input_image()
                rospy.loginfo_throttle(0.1, f"Goal projected to image at ({x}, {y})")
            else:
                rospy.loginfo_throttle(0.1, f"Goal is outside of image, rotating!")
                self.handle_goal_out_of_sight()
                
        except Exception as e:
            rospy.logerr(f"Error in goal projection: {e}")
    
    def transform_goal_to_camera_frame(self):
        """Transform the 3D goal from its frame to the camera frame"""
        try:
            # Get transform from goal frame to camera frame
            transform = self.tf_buffer.lookup_transform(
                self.camera_info.header.frame_id,
                self.current_goal.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.5)
            )
            
            # Transform the goal pose
            goal_in_camera = tf2_geometry_msgs.do_transform_pose(
                self.current_goal, transform
            )
            
            return goal_in_camera.pose.position
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error in goal transform: {e}")
            return None
    
    def project_point_to_image(self, point_3d):
        """Project a 3D point to 2D image coordinates using camera intrinsics"""
        if point_3d.z <= 0:
            # Point is behind camera
            return -1, -1
            
        x = point_3d.x / point_3d.z
        y = point_3d.y / point_3d.z
        
        # Apply camera intrinsics
        x_pixel = x * self.projector.fx + self.projector.cx
        y_pixel = y * self.projector.fy + self.projector.cy
        
        return int(x_pixel), int(y_pixel)
    
    def publish_goal_2d(self, x, y):
        """Publish 2D goal coordinates for the VLM inference node"""
        goal_2d_msg = Float32MultiArray()
        goal_2d_msg.data = [float(x), float(y)]
        self.goal_2d_pub.publish(goal_2d_msg)
    
    def publish_input_image(self):
        """Publish the current image for VLM processing"""
        if self.current_image is not None:
            img_msg = self.cv_bridge.cv2_to_imgmsg(self.current_image, "bgr8")
            img_msg.header.stamp = self.image_timestamp
            img_msg.header.frame_id = self.camera_info.header.frame_id if self.camera_info else "camera"
            self.vlm_input_image_pub.publish(img_msg)
    
    def handle_goal_out_of_sight(self):
        """Rotate the robot toward the goal when it's not in camera view"""
        if not self.goal_in_sight and self.current_goal and self.current_odom:
            # Calculate the angle to the goal
            goal_pos = self.current_goal.pose.position
            robot_pos = self.current_odom.pose.pose.position
            robot_quat = [
                self.current_odom.pose.pose.orientation.x,
                self.current_odom.pose.pose.orientation.y,
                self.current_odom.pose.pose.orientation.z,
                self.current_odom.pose.pose.orientation.w
            ]
            _, _, robot_yaw = euler_from_quaternion(robot_quat)
            
            # Calculate angle to goal
            goal_angle = np.arctan2(
                goal_pos.y - robot_pos.y,
                goal_pos.x - robot_pos.x
            )
            
            # Calculate the angular difference
            angle_diff = (goal_angle - robot_yaw + np.pi) % (2 * np.pi) - np.pi
            
            # Create twist command to rotate toward goal
            twist = Twist()
            twist.angular.z = np.sign(angle_diff) * min(abs(angle_diff), self.rotation_speed)
            self.cmd_vel_pub.publish(twist)
            
            rospy.loginfo_throttle(1.0, "Goal not in camera view, rotating towards it")
    
    def process_waypoint_tracking(self):
        """Track and navigate to waypoints"""
        if not self.current_odom:
            return
            
        # If no waypoints are available and goal is not in sight, keep rotating
        if not self.waypoints_3d:
            rospy.loginfo_throttle(0.1, f"Waypoints not available")
            if not self.goal_in_sight and self.current_goal:
                self.handle_goal_out_of_sight()
            return
        
        rospy.loginfo("Goal in sight!")

        # If we have waypoints, track the current one
        if self.current_waypoint_index < len(self.waypoints_3d):
            current_waypoint = self.waypoints_3d[self.current_waypoint_index]
            
            # Calculate distance to waypoint
            robot_pos = self.current_odom.pose.pose.position
            waypoint_pos = current_waypoint.pose.position
            distance = np.sqrt(
                (waypoint_pos.x - robot_pos.x)**2 + 
                (waypoint_pos.y - robot_pos.y)**2
            )
            
            if distance < self.waypoint_distance_threshold:
                # Move to the next waypoint
                self.current_waypoint_index += 1
                rospy.loginfo(f"Waypoint {self.current_waypoint_index-1} reached, moving to next")
                
                # Publish visualization if enabled
                if self.visualization_enabled:
                    self.publish_current_waypoint_visualization()
                
                if self.current_waypoint_index < len(self.waypoints_3d):
                    self.compute_and_publish_velocity()
                else:
                    # End of path, stop the robot
                    self.cmd_vel_pub.publish(Twist())
                    rospy.loginfo("End of path reached, stopping robot")
            else:
                # Keep tracking the current waypoint
                self.compute_and_publish_velocity()
                
                # Publish visualization if enabled
                if self.visualization_enabled:
                    self.publish_current_waypoint_visualization()
    
    def compute_and_publish_velocity(self):
        """Compute and publish velocity commands to follow the current waypoint"""
        if self.current_waypoint_index >= len(self.waypoints_3d):
            return
            
        try:
            current_waypoint = self.waypoints_3d[self.current_waypoint_index]
            
            # Transform waypoint to base_link frame
            waypoint_base = tf2_geometry_msgs.do_transform_pose(
                current_waypoint,
                self.tf_buffer.lookup_transform(
                    "base_link",
                    current_waypoint.header.frame_id,
                    rospy.Time(0),
                    rospy.Duration(0.5)
                )
            )
            
            # Extract position from the transformed waypoint
            diff = np.array([
                waypoint_base.pose.position.x,
                waypoint_base.pose.position.y,
            ])
            target_heading = np.arctan2(diff[1], diff[0])
            
            # Create velocity command using P controller (similar to carrot_picker)
            vel_msg = Twist()
            
            # Calculate linear velocities
            vel_msg.linear.x = np.clip(self.p_lin_x * diff[0], -self.max_x_vel, self.max_x_vel)
            vel_msg.linear.y = np.clip(self.p_lin_y * diff[1], -self.max_y_vel, self.max_y_vel)
            
            # Calculate angular velocity to track the heading
            angle_diff = (target_heading + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            vel_msg.angular.z = np.clip(self.p_ang * angle_diff, -self.max_angular_vel, self.max_angular_vel)
            
            # Don't walk backwards and turn
            if np.abs(angle_diff) > np.radians(self.fwd_angle_thresh/2.0):
                vel_msg.linear.x = 0
                vel_msg.linear.y = 0
            
            # Don't shake at goal
            if np.linalg.norm(diff) < self.carrot_radius:
                vel_msg.linear.x = 0
                vel_msg.linear.y = 0
                vel_msg.angular.z = 0
            
            # Publish the velocity command
            self.cmd_vel_pub.publish(vel_msg)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error in velocity computation: {e}")
            # If transform fails, stop the robot
            self.cmd_vel_pub.publish(Twist())
    
    def publish_waypoint_visualization(self):
        """Publish marker array for visualizing all waypoints"""
        if not self.visualization_enabled or not self.waypoints_3d:
            return
            
        marker_array = MarkerArray()
        
        for i, waypoint in enumerate(self.waypoints_3d):
            marker = Marker()
            marker.header = waypoint.header
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position = waypoint.pose.position
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # Color based on index (red to green gradient)
            ratio = float(i) / max(1, len(self.waypoints_3d) - 1)
            marker.color.r = 1.0 - ratio
            marker.color.g = ratio
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)
            
            # Add a text marker with the waypoint number
            text_marker = Marker()
            text_marker.header = waypoint.header
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.id = i + 1000  # Offset to avoid ID collision
            text_marker.pose.position = waypoint.pose.position
            text_marker.pose.position.z += 0.3  # Place text above sphere
            text_marker.text = str(i)
            text_marker.scale.z = 0.2  # Text height
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8
            
            marker_array.markers.append(text_marker)
        
        self.waypoints_pub.publish(marker_array)
    
    def publish_current_waypoint_visualization(self):
        """Publish marker for the current active waypoint"""
        if not self.visualization_enabled or not self.waypoints_3d or self.current_waypoint_index >= len(self.waypoints_3d):
            return
            
        waypoint = self.waypoints_3d[self.current_waypoint_index]
        marker = Marker()
        marker.header = waypoint.header
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 0
        marker.pose.position = waypoint.pose.position
        marker.pose.orientation = waypoint.pose.orientation
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.current_waypoint_pub.publish(marker)
    
    def publish_goal_visualization(self):
        """Publish marker for the goal position"""
        if not self.visualization_enabled or not self.current_goal:
            return
            
        marker = Marker()
        marker.header = self.current_goal.header
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = 0
        marker.pose = self.current_goal.pose
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.1  # Arrow width
        marker.scale.z = 0.1  # Arrow height
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        self.goal_marker_pub.publish(marker)
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)  # 10 Hz control loop
        
        while not rospy.is_shutdown():
            # Republish visualizations periodically
            if self.visualization_enabled:
                if self.waypoints_3d:
                    self.publish_waypoint_visualization()
                    self.publish_current_waypoint_visualization()
                if self.current_goal:
                    self.publish_goal_visualization()
                
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("vlm_navigation_node")
        node = VLMNavigationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass