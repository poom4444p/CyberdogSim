#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
import message_filters
import threading
import numpy as np
from queue import Queue
from trajectory_projection import Projector

class MultiPointClicker:
    def __init__(self):
        rospy.init_node('multi_point_clicker')

        self.bridge = CvBridge()
        self.points = []
        self.max_points = 5

        self.rgb_image = None
        self.depth_image = None
        self.rgb_info = None

        self.image_lock = threading.Lock()
        self.running = True

        self.image_queue = Queue(maxsize=1)

        self.projector = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rgb_sub = message_filters.Subscriber('/zed2/zed_node/left/image_rect_color', Image)
        depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
        rgb_info_sub = message_filters.Subscriber('/zed2/zed_node/left/camera_info', CameraInfo)

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, rgb_info_sub], queue_size=10, slop=0.1
        )
        ts.registerCallback(self.image_callback)

        self.marker_pub = rospy.Publisher('/clicked_points_markers', MarkerArray, queue_size=1)

        self.window_name = 'Multi-Point Clicker'

        rospy.on_shutdown(self.shutdown_hook)

        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def shutdown_hook(self):
        self.running = False
        self.display_thread.join()
        cv2.destroyAllWindows()

    def image_callback(self, rgb_msg, depth_msg, rgb_info_msg):
        with self.image_lock:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            self.rgb_info = rgb_info_msg

            if self.projector is None:
                self.projector = Projector(K=self.rgb_info.K)

            display_img = self.rgb_image.copy()
            for pt in self.points:
                cv2.circle(display_img, pt, 5, (0, 255, 0), -1)

            if not self.image_queue.empty():
                self.image_queue.get_nowait()
            self.image_queue.put_nowait(display_img)

    def display_loop(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while self.running and not rospy.is_shutdown():
            try:
                image = self.image_queue.get(timeout=0.1)
                cv2.imshow(self.window_name, image)
                key = cv2.waitKey(1)
                if key == 27:
                    rospy.signal_shutdown("User pressed ESC")
                    break
            except:
                continue

    def mouse_callback(self, event, x, y, flags, param):
        process_now = False

        if event == cv2.EVENT_LBUTTONDOWN:
            with self.image_lock:
                if len(self.points) < self.max_points and self.rgb_image is not None:
                    self.points.append((x, y))
                    rospy.loginfo(f"Recorded point {len(self.points)}: ({x}, {y})")

                if len(self.points) == self.max_points:
                    process_now = True

        if process_now:
            self.process_points()

    def process_points(self):
        with self.image_lock:
            if self.rgb_image is None or self.depth_image is None:
                rospy.logwarn("Images not ready yet.")
                return

            local_points = self.points.copy()
            self.points.clear()

        points_3d = self.projector.get_3d_points(local_points, self.depth_image)
        print(f"Projected 3D points: {points_3d}")

        marker_array = MarkerArray()
        for idx, point in enumerate(points_3d):
            marker = Marker()
            marker.header.frame_id = self.rgb_info.header.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.id = idx
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        rospy.loginfo(f"Published MarkerArray with {len(points_3d)} points")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        clicker = MultiPointClicker()
        clicker.run()
    except rospy.ROSInterruptException:
        pass
