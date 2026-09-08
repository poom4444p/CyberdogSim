#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class VLMInferenceNode:
    def __init__(self):
        rospy.init_node("vlm_inference_dummy")

        self.cv_bridge = CvBridge()

        self.goal_2d = None
        self.latest_image = None
        self.image_height = 0
        self.image_width = 0

        self.path_pub = rospy.Publisher('/vamos/path_2d', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/vamos/goal_2d', Float32MultiArray, self.goal_callback)
        rospy.Subscriber('/vamos/input_image', Image, self.image_callback)

        rospy.loginfo("VLM Inference Dummy Node initialized")

    def goal_callback(self, msg):
        self.goal_2d = msg.data
        rospy.loginfo(f"Received goal: ({self.goal_2d[0]}, {self.goal_2d[1]})")
        self.generate_path()

    def image_callback(self, msg):
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_height, self.image_width = self.latest_image.shape[:2]
            rospy.loginfo_throttle(5.0, f"Received image: {self.image_width}x{self.image_height}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def generate_path(self):
        if self.goal_2d is None or self.image_height == 0:
            rospy.logwarn("Missing goal or image data, cannot generate data")
            return
    
        start_x = self.image_width / 2
        start_y = self.image_height - 1

        goal_x, goal_y = self.goal_2d

        num_points = 5
        path_points = []

        t_values = np.linspace(0, 1, num_points+1)
        t_values = t_values[1:]
        for t in t_values:
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)
            path_points.extend([x, y])
        
        path_msg = Float32MultiArray()
        path_msg.data = path_points
        self.path_pub.publish(path_msg)

        rospy.loginfo(f"Published path with {num_points} points from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")

    def run(self):
        rate = rospy.Rate(1)
        rospy.loginfo("VLM Inference Dummy Node running")

        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == "__main__":
    try:
        node = VLMInferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass