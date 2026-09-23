#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import numpy as np
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
import message_filters
import threading
from queue import Queue

class PointClicker:
    def __init__(self):
        rospy.init_node('point_clicker')

        self.bridge = CvBridge()

        self.rgb_image = None
        self.depth_image = None
        self.rgb_info = None
        self.depth_info = None

        self.window_created = False

        # Image display queue and lock
        self.image_queue = Queue(maxsize=1)
        self.image_lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Set up subscribers
        self.rgb_sub = message_filters.Subscriber(
            '/zed2/zed_node/left/image_rect_color',
            Image
        )

        self.rgb_info_sub = message_filters.Subscriber(
            '/zed2/zed_node/left/camera_info',
            CameraInfo
        )

        self.depth_sub = message_filters.Subscriber(
            '/zed2/zed_node/depth/depth_registered',
            Image
        )

        self.depth_info_sub = message_filters.Subscriber(
            '/zed2/zed_node/depth/camera_info',
            CameraInfo
        )

        # Synchronize the subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.rgb_info_sub, self.depth_info_sub],
            queue_size=10,
            slop=0.1
        )  ## TODO: Do I need the rgb_info and depth_info here? They should be loaded once and then ignored
        self.ts.registerCallback(self.sync_callback)

        self.point_pub = rospy.Publisher(
            '/clicked_point_3d',
            PointStamped,
            queue_size=1
        )

        self.point2d_pub = rospy.Publisher(
            '/clicked_point_2d',
            PointStamped,
            queue_size=1
        )

        self.image_pub = rospy.Publisher(
            '/clicked_image',
            Image,
            queue_size=1
        )

        self.depth_image_pub = rospy.Publisher(
            '/clicked_depth_image',
            Image,
            queue_size=1
        )

        self.camera_info_pub = rospy.Publisher(
            '/clicked_camera_info',
            CameraInfo, 
            queue_size=1
        )

        self.window_name = 'RGB Image'
        # cv2.namedWindow(self.window_name)
        # cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Start display thread
        self.running = True
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self):
        self.running = False
        self.display_thread.join()
        cv2.destroyAllWindows()

    def sync_callback(self, rgb_msg, depth_msg, rgb_info_msg, depth_info_msg):
        self.rgb_info = rgb_info_msg
        self.depth_info = depth_info_msg

        # Process RGB image
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # Process depth image
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        rospy.loginfo_throttle(5.0, f"Depth states:\n\t- min: {np.nanmin(depth_image):.3f} m\n\t- max: {np.nanmax(depth_image):.3f} m\n\t- mean: {np.nanmean(depth_image):.3f} m")

        if rgb_image is not None and depth_image is not None:
            with self.image_lock:
                self.rgb_image = rgb_image.copy()
                self.depth_image = depth_image.copy()

                # Update display queue
                if not self.image_queue.empty():
                    try:
                        self.image_queue.get_nowait()
                    except Queue.Empty:
                        pass
                try:
                    self.image_queue.put_nowait(rgb_image.copy())
                except Queue.Full:
                    pass

        self.image_pub.publish(rgb_msg)  # Original RGB image
        self.depth_image_pub.publish(depth_msg)  # Corresponding depth image
        self.camera_info_pub.publish(rgb_info_msg)  # RGB camera info

    def create_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.window_created = True

    def display_loop(self):
        '''Separate thread for displaying images'''
        self.create_window()

        while self.running and not rospy.is_shutdown():
            with self.image_lock:
                if self.rgb_image is not None:
                    cv2.imshow(self.window_name, self.rgb_image)

            if cv2.waitKey(30) & 0xFF == 27:
                rospy.signal_shutdown("User pressed ESC")
                break
            # 
            # try:
                # image = self.image_queue.get(timeout=0.1)
                # cv2.imshow(self.window_name, image)
                # cv2.waitKey(1)
            # except:
                # continue

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Registered a click!")
            with self.image_lock:
                if all(msg is not None for msg in [self.rgb_image, self.depth_image, self.rgb_info, self.depth_info]):
                    self.process_click(x, y)

    def process_click(self, x, y):
        depth_meters = self.depth_image[y, x]

        if np.isnan(depth_meters) or depth_meters == 0:
            rospy.logwarn(f"Invalid depth at clicked point: ({x}, {y})")
            return

        # Publish current RGB image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(self.rgb_image, encoding='bgr8')
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = self.rgb_info.header.frame_id
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish image: {e}")
        
        height, width = self.rgb_image.shape[:2]
        norm_x = float(x) / width
        norm_y = float(y) / height

        point2d_msg = PointStamped()
        point2d_msg.header.frame_id = self.rgb_info.header.frame_id
        point2d_msg.header.stamp = rospy.Time.now()
        point2d_msg.point.x = norm_x
        point2d_msg.point.y = norm_y
        point2d_msg.point.z = 0.0
        self.point2d_pub.publish(point2d_msg)

        # Get camera intrinsics from RGB camera info

        K = np.array(self.rgb_info.K).reshape(3, 3)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Calculate 3D point in camera frame
        x_cam = (x - cx) * depth_meters / fx
        y_cam = (y - cy) * depth_meters / fy
        z_cam = depth_meters

        # Create PointStamped message in camera_frame
        point_camera = PointStamped()
        point_camera.header.frame_id = self.rgb_info.header.frame_id
        point_camera.header.stamp = rospy.Time.now()
        point_camera.point.x = x_cam
        point_camera.point.y = y_cam
        point_camera.point.z = z_cam

        try:
            point_base = self.tf_buffer.transform(point_camera, "base_link", rospy.Duration(1.0))
            self.point_pub.publish(point_base)

            rospy.loginfo(f"Raw depth at point: {depth_meters:.3f} meters")
            rospy.loginfo(f"3D point in base frame: x={point_base.point.x:.3f}, y={point_base.point.y:.3f}, z={point_base.point.z:.3f}")

        except (tf2_ros.LookupException, tf2_ros.ConnectorException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform failed: {e}")

        
    def run(self):
        rospy.spin()
        # while not rospy.is_shutdown():
        #     if self.rgb_image is not None:
        #         cv2.imshow(self.window_name, self.rgb_image)
        #         cv2.waitKey(1)
        #     rospy.sleep(0.1)

if __name__ == "__main__":
    try:
        clicker = PointClicker()
        clicker.run()
    except rospy.ROSInterruptException:
        pass
