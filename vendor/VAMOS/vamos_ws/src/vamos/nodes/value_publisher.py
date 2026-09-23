#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray
from ValueProcessor import ValueProcessor
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ValuePublisher:
    def __init__(self, imu_topic, heightmap_topic, nn_path):
        # Initialize the ROS node
        rospy.init_node('value_publisher')

        self.imu_data = None
        self.heightmap = None
        self.goal_markers = None

        self.value_model = ValueProcessor(nn_path, map_out=True)

        # Subscribers
        rospy.Subscriber(imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(heightmap_topic, Float32MultiArray, self.heightmap_callback)
        
        # Publisher
        value_map_topic = rospy.get_param('~value_map_topic', '/vamos/value_map')
        value_map_image_topic = rospy.get_param('~value_map_image_topic', '/vamos/value_map_image')
        self.value_pub = rospy.Publisher(value_map_topic, GridMap, queue_size=1)
        self.image_pub = rospy.Publisher(value_map_image_topic, Image, queue_size=1)
        self.bridge = CvBridge()
        # Run processing in a timer callback
        rospy.Timer(rospy.Duration(0.1), self.process_and_publish)

    def imu_callback(self, msg):
        self.imu_data = msg

    def heightmap_callback(self, msg):
        self.heightmap = msg

    def process_and_publish(self, event):
        # Ensure all messages have been received at least once
        if self.imu_data is None or self.heightmap is None:
            return

        # Extract orientation from IMU (convert quaternion to Euler angles)
        quaternion = np.array([
            self.imu_data.orientation.x,
            self.imu_data.orientation.y,
            self.imu_data.orientation.z,
            self.imu_data.orientation.w
        ])
        heightmap_array = np.array(self.heightmap.data)
        heightmap = heightmap_array.reshape((1,21,21)).astype(np.float32)
        values = self.value_model.inference(heightmap)
        
        rows, cols = values.shape
        msg = GridMap()
        robot_frame = rospy.get_param('~robot_frame', 'base_link')
        msg.info.header = Header(stamp=rospy.Time.now(),
                        frame_id=robot_frame)
        resolution = 0.25
        msg.info.resolution = resolution
        msg.info.length_x = cols * resolution 
        msg.info.length_y = rows * resolution

        msg.info.pose.position = Point(msg.info.length_x / 2.0,
                              msg.info.length_y / 2.0,
                              0.0)
        msg.info.pose.orientation.w = 1.0  # self._map_q.w
        msg.info.pose.orientation.x = 0.0  # self._map_q.x
        msg.info.pose.orientation.y = 0.0  # self._map_q.y
        msg.info.pose.orientation.z = 0.0  # self._map_q.z
        msg.layers = ["forward_value"]
        flat = values.flatten(order='C').tolist()

        layout = MAL()
        dim0 = MAD(label="column_index", size=rows, stride=rows * cols)
        dim1 = MAD(label="row_index",  size=cols, stride=cols)
        layout.dim = [dim0, dim1]
        layout.data_offset = 0
        arr = Float32MultiArray()
        arr.layout = layout
        arr.data = flat
        msg.data = [arr]  

        values = np.flipud(values)

        img_msg = self.bridge.cv2_to_imgmsg(values, encoding="32FC1")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "base_link"

        self.image_pub.publish(img_msg)

        self.value_pub.publish(msg)

if __name__ == '__main__':
    try:
        imu_topic = rospy.get_param('~imu_topic', '/mavros/imu/data')
        heightmap_topic = rospy.get_param('~heightmap_topic', '/heights_flat_array')
        nn_path = rospy.get_param('~nn_path', '/home/dino/spot/spot_models/value_functions/rough_critic.onnx')
        ValuePublisher(imu_topic, heightmap_topic, nn_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass