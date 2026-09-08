#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from collections import deque
from nav_msgs.msg import Odometry  # Add this import
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class WaypointManager:
    def __init__(self):
        rospy.init_node('waypoint_manager')
        
        # Parameters
        self.distance_threshold = rospy.get_param('~distance_threshold', 0.2)  # meters
        self.allow_replan = rospy.get_param('~allow_replan', True)
        
        # Initialize waypoint queue and current target
        self.waypoints = deque()
        self.current_waypoint = None
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        self.path_sub = rospy.Subscriber(
            '/output_path',
            Path,
            self.path_callback,
            queue_size=1
        )
        
        self.pose_sub = rospy.Subscriber(
            '/spot/odometry_corrected', 
            Odometry,
            self.pose_callback,
            queue_size=1
        )
        
        # Publisher for current target waypoint
        self.target_pub = rospy.Publisher(
            '/current_waypoint',
            PoseStamped,
            queue_size=1
        )
        
        # Visualization parameters
        self.enable_viz = rospy.get_param('~enable_viz', True)
        self.completed_waypoints = []
        
        if self.enable_viz:
            self.marker_pub = rospy.Publisher(
                '/waypoint_markers',
                MarkerArray,
                queue_size=1
            )
            
            # Marker colors
            self.future_color = ColorRGBA(0.0, 1.0, 0.0, 1.0)    # Green
            self.current_color = ColorRGBA(1.0, 0.0, 0.0, 1.0)   # Red
            self.completed_color = ColorRGBA(0.5, 0.5, 0.5, 0.5) # Gray

        rospy.loginfo("Waypoint Manager initialized")

    def path_callback(self, path_msg):
        """Handle new path messages"""
        if not self.allow_replan:
            rospy.loginfo("Ignoring new path - replanning disabled")
            return
            
        try:
            # Transform waypoints to vision frame
            transformed_poses = []
            for pose in path_msg.poses:
                # Transform from base_link to vision frame
                transformed = self.tf_buffer.transform(pose, "vision", rospy.Duration(1.0))
                transformed_poses.append(transformed)
            
            # Update waypoint queue
            self.waypoints = deque(transformed_poses)
            if len(self.waypoints) > 0:
                self.completed_waypoints = []  # Reset completed waypoints for new path
                self.current_waypoint = self.waypoints.popleft()
                self.target_pub.publish(self.current_waypoint)
                self.publish_markers()
                rospy.loginfo(f"New path received with {len(transformed_poses)} waypoints")
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform failed: {e}")

    def pose_callback(self, odom_msg):
        """Handle robot pose updates from odometry"""
        if not self.current_waypoint:
            return
            
        try:
            # Create PoseStamped from Odometry
            pose_stamped = PoseStamped()
            pose_stamped.header = odom_msg.header
            pose_stamped.pose = odom_msg.pose.pose
            
            # Transform robot pose to vision frame
            robot_pose = self.tf_buffer.transform(pose_stamped, "vision", rospy.Duration(700.0))
            
            # Calculate distance to current waypoint
            distance = self.calculate_distance(robot_pose, self.current_waypoint)
            rospy.loginfo(f"Distance to closest waypoint: {distance}")
            
            if distance < self.distance_threshold:
                rospy.loginfo(f"Waypoint reached! Distance: {distance:.2f}m")
                self.update_current_waypoint()
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform failed: {e}")

    def calculate_distance(self, pose1, pose2, only_2d=True):
        """Calculate Euclidean distance between two poses"""
        p1 = pose1.pose.position
        p2 = pose2.pose.position
        if only_2d:
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)    
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def update_current_waypoint(self):
        """Update current waypoint to next in queue"""
        if self.current_waypoint:
            self.completed_waypoints.append(self.current_waypoint)
            
        if len(self.waypoints) > 0:
            self.current_waypoint = self.waypoints.popleft()
            self.target_pub.publish(self.current_waypoint)
            rospy.loginfo(f"Moving to next waypoint. {len(self.waypoints)} remaining")
        else:
            self.current_waypoint = None
            rospy.loginfo("Path completed!")
        
        self.publish_markers()

    def create_waypoint_marker(self, pose, id, color, is_completed=False):
        marker = Marker()
        marker.header.frame_id = "vision"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = id
        marker.type = Marker.SPHERE if not is_completed else Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color = color
        marker.lifetime = rospy.Duration(0)
        return marker

    def publish_markers(self):
        if not self.enable_viz:
            return
            
        marker_array = MarkerArray()
        
        # Add completed waypoints
        for idx, wp in enumerate(self.completed_waypoints):
            marker = self.create_waypoint_marker(
                wp, idx, self.completed_color, is_completed=True
            )
            marker_array.markers.append(marker)
        
        # Add current waypoint
        if self.current_waypoint:
            marker = self.create_waypoint_marker(
                self.current_waypoint,
                len(self.completed_waypoints),
                self.current_color
            )
            marker_array.markers.append(marker)
        
        # Add future waypoints
        for idx, wp in enumerate(self.waypoints):
            marker = self.create_waypoint_marker(
                wp,
                len(self.completed_waypoints) + 1 + idx,
                self.future_color
            )
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = WaypointManager()
        node.run()
    except rospy.ROSInterruptException:
        pass