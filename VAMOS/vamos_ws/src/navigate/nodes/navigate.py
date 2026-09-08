#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
# from cv_bridge import CvBridge
import message_filters
import threading
import time
from queue import Queue, Empty, Full
from trajectory_prediction import Packet, TrajectoryPredictorBase, DummyTrajectoryPredictor
from vamos import VLMTrajectoryPredictor

PREDICTOR_MAP = {
    "base": TrajectoryPredictorBase,
    "dummy": DummyTrajectoryPredictor,
    "vlm": VLMTrajectoryPredictor,
}

def image_msg_to_rgb8(msg):
    """
    Convert a sensor_msgs/Image with 8-bit/color encoding into an H×W×3 uint8 RGB array.
    """
    height = msg.height
    width  = msg.width
    row_bytes = msg.step
    bpp = row_bytes // width

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape((height, row_bytes))
    arr = arr[:, : width * bpp]
    arr = arr.reshape((height, width, bpp))

    # drop any alpha channel
    if bpp == 4:
        arr = arr[:, :, :3]
    elif bpp != 3:
        raise ValueError(f"Unsupported bytes_per_pixel={bpp}")

    # Now arr is in BGR order—swap to RGB:
    return arr[..., ::-1]

def image_msg_to_depth32(msg):
    arr = np.frombuffer(msg.data, dtype=np.float32)
    return arr.reshape(msg.height, msg.width)

class Navigation2dNode:
    def __init__(self, node_handle):
        self.node_handle = node_handle

        # # Topics should now point at your compressed & throttled feeds:
        self.image_topic = rospy.get_param(
            "~image_topic",
            "/zed2i/zed_node/left/image_rect_color/compressed_throttle"
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic",
            "/zed2i/zed_node/depth/depth_registered/compressedDepth_throttle"
        )
        self.depth_camera_info_topic = rospy.get_param(
            "~depth_camera_info_topic",
            "/zed2i/zed_node/depth/camera_info"
        )
        self.goal_topic = rospy.get_param("~goal_topic", "/spot/goal")

        print(f"Subscribing to image topic: {self.image_topic}")
        print(f"Subscribing to depth topic: {self.depth_topic}")
        print(f"Subscribing to depth camera info topic: {self.depth_camera_info_topic}")
        print(f"Subscribing to goal topic: {self.goal_topic}")

        # Add subscriber to value map topic
        self.value_map_topic = rospy.get_param('~value_map_image_topic_trajectories', '/vamos/value_map_image')

        self.predictor_type = rospy.get_param("~predictor_type", "dummy")
        self.time_threshold = rospy.get_param("~time_threshold", 1000)

        self._inference_queue = Queue(maxsize=1)
        self._worker_thread = None 

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # # Debug subscribers
        if rospy.get_param("~debug_subscribers", False):
            self.image_sub_debug = rospy.Subscriber(
                self.image_topic,
                Image,
                self.image_debug_callback
            )
            self.depth_sub_debug = rospy.Subscriber(
                self.depth_topic,
                Image,
                self.depth_debug_callback
            )
            self.depth_info_sub_debug = rospy.Subscriber(
                self.depth_camera_info_topic,
                CameraInfo,
                self.info_debug_callback
            )
            self.goal_sub_debug = rospy.Subscriber(
                self.goal_topic,
                PoseStamped,
                self.goal_debug_callback
            )
            self.value_sub_debug = rospy.Subscriber(
                self.value_map_topic,
                Image,
                self.value_debug_callback
            )

        # Cache latest CameraInfo
        self.latest_info = None
        rospy.Subscriber(
            self.depth_camera_info_topic,
            CameraInfo,
            self._update_camera_info
        )

        # message_filters subscribers
        self.image_sub = message_filters.Subscriber(
            self.image_topic,
            Image
        )
        self.depth_sub = message_filters.Subscriber(
            self.depth_topic,
            Image
        )
        self.goal_sub = message_filters.Subscriber(
            self.goal_topic,
            PoseStamped
        )

        self.value_map_image_sub = message_filters.Subscriber(
            self.value_map_topic, 
            Image
        )

        # Synchronizer WITHOUT camera_info
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.goal_sub, self.value_map_image_sub],
            queue_size=10,
            slop=0.7,
            allow_headerless=True
        )
        self.ts.registerCallback(self.sync_callback)

        # Pick predictor
        cls = PREDICTOR_MAP.get(self.predictor_type)
        if cls:
            self.trajectory_predictor = cls(node_handle, self.tf_buffer, self.time_threshold)
        else:
            raise ValueError(f"Unknown predictor type {self.predictor_type!r}")
        
        # Start the worker thread
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()

        # Register shutdown hook
        rospy.on_shutdown(self._shutdown)

        rospy.loginfo("Navigation 2D Node Initialized with queue-based worker")

    # Debug callbacks unchanged
    def image_debug_callback(self, msg: CompressedImage):
        rospy.loginfo(f"Received RGB Image (stamp: {msg.header.stamp.to_sec()})")

    def depth_debug_callback(self, msg: Image):
        rospy.loginfo(f"Received Depth Image (stamp: {msg.header.stamp.to_sec()})")

    def info_debug_callback(self, msg: CameraInfo):
        rospy.loginfo(f"Received CameraInfo (stamp: {msg.header.stamp.to_sec()})")

    def goal_debug_callback(self, msg: PoseStamped):
        rospy.loginfo(f"Received Goal (stamp: {msg.header.stamp.to_sec()})")

    def value_debug_callback(self, msg: Image):
        rospy.loginfo(f"Received Value Map Image (stamp: {msg.header.stamp.to_sec()})")

    def _update_camera_info(self, msg: CameraInfo):
        self.latest_info = msg


    def sync_callback(self, image_msg, depth_msg, goal_msg, value_map_msg):
        start_cb_time = time.time()

        # Ensure we have camera info
        if self.latest_info is None:
            rospy.logdebug("No CameraInfo received yet, skipping packet")
            return

        rospy.loginfo("Received packet!")

        # Decode compressed image → BGR8 numpy array
        image_data = image_msg_to_rgb8(image_msg)

        # Decode raw Depth → float32 depth image
        depth_data = image_msg_to_depth32(depth_msg)

        # Decode value map image
        value_map_data = image_msg_to_depth32(value_map_msg)

        # Camera intrinsics from cached info
        depth_K = np.array(self.latest_info.K).reshape(3, 3)
        
        # TF for goal in camera optical frame
        try:
            t = self.tf_buffer.lookup_transform(
                self.latest_info.header.frame_id,  # zed2i_left_camera_optical_frame
                goal_msg.header.frame_id,  # vision
                goal_msg.header.stamp,
                rospy.Duration(0.1)
            )
            # import ipdb;ipdb.set_trace()
            goal_in_cam = tf2_geometry_msgs.do_transform_pose(goal_msg, t)
            p = goal_in_cam.pose.position
            goal_3d = np.array([[p.x, p.y, p.z]])
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        # Secondary TF (unchanged)
        try:
            t2 = self.tf_buffer.lookup_transform(
                goal_msg.header.frame_id, # vision
                depth_msg.header.frame_id,  # zed2i_left_camera_optical_frame
                depth_msg.header.stamp,
                rospy.Duration(0.1),
            )
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        robot_pose_vision = None
        try:
            # Use same timestamp as the depth image to keep it consistent
            common_stamp = depth_msg.header.stamp
            robot_transform_vision_to_base = self.tf_buffer.lookup_transform(
                goal_msg.header.frame_id,  # Target frame
                rospy.get_param("~robot_frame", "base_link"),  # Source frame
                common_stamp,  # Timestamp
                rospy.Duration(0.1),  # Timeout
            )

            # Convert TransformStamped to PoseStamped
            robot_pose_vision = PoseStamped()
            robot_pose_vision.header.stamp = common_stamp
            robot_pose_vision.header.frame_id = goal_msg.header.frame_id # TODO: Check 
            robot_pose_vision.pose.position.x = robot_transform_vision_to_base.transform.translation.x
            robot_pose_vision.pose.position.y = robot_transform_vision_to_base.transform.translation.y
            robot_pose_vision.pose.position.z = robot_transform_vision_to_base.transform.translation.z
            robot_pose_vision.pose.orientation = robot_transform_vision_to_base.transform.rotation

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            robot_frame = rospy.get_param("~robot_frame", "base_link")
            rospy.logwarn_throttle(2.0, f"TF lookup {goal_msg.header.frame_id} -> {robot_frame} failed: {e}. Packet will lack robot pose.")

        ts = depth_msg.header.stamp.to_sec()

        current_packet = Packet(
            image=image_data,
            depth_image=depth_data,
            depth_intrinsics=depth_K,
            camera_optical_to_vision_transform=t2,
            goal_3d=goal_3d,
            goal_2d=None,
            robot_pose_vision=robot_pose_vision,
            value_map=value_map_data,
            timestamp=ts
        )

        try:
            self._inference_queue.put_nowait(current_packet)
            rospy.logdebug("Packet added to queue")
        except Full:
            # If the queue is full (worker hasn't picked up the last one),
            # discard the old one and put the new one.
            rospy.logdebug("Inference queue full. Discarding old packet and adding new one.")
            try:
                _ = self._inference_queue.get_nowait() # Discard previous packet
            except Empty:
                pass # Should not happen if Full was raised, but safeguard
            try:
                self._inference_queue.put_nowait(current_packet) # Put the newest packet
            except Full:
                # Should really not happen after the get_nowait, log if it does
                rospy.logdebug("Failed to put packet in queue even after clearing.")

        cb_duration = time.time() - start_cb_time

    def _inference_loop(self):
        """Continuously fetches packets from the queue and processes them."""
        rospy.logdebug("Inference worker thread started.")
        while not rospy.is_shutdown():
            packet_to_process = None
            try:
                # Block waiting for a packet, but with a timeout
                # to allow checking rospy.is_shutdown() periodically.
                packet_to_process = self._inference_queue.get(timeout=0.5) # Timeout in seconds
                rospy.logdebug(f"Worker dequeued packet ts: {packet_to_process.timestamp}")

            except Empty:
                # Queue was empty during the timeout period, just loop again
                continue
            except Exception as e:
                # Handle potential issues with the queue itself
                rospy.logerr(f"Error getting packet from queue: {e}", exc_info=True)
                time.sleep(0.1) # Avoid tight loop on queue errors
                continue

            # Process the packet if successfully retrieved
            if packet_to_process:
                try:
                    start_process_time = time.time()
                    # The actual workhorse - calls VLM or other predictor logic
                    self.trajectory_predictor.process_input(packet_to_process)
                    process_duration = time.time() - start_process_time
                    rospy.logdebug(f"Worker finished processing in {process_duration:.3f} s")

                except Exception as e:
                    # Catch errors from process_input to prevent worker crash
                    rospy.logerr(f"Exception in worker thread during process_input: {e}", exc_info=True)
            else:
                 # This case shouldn't be reached if get() succeeded without Empty
                 rospy.logdebug("Worker loop: packet_to_process is None after get().")

        rospy.loginfo("Inference worker thread stopping.")


    def run(self):
        """Main loop: Now just keeps the node alive for callbacks and worker."""
        rospy.loginfo("Navigation node spinning...")
        # rospy.spin() handles callbacks and prevents the main thread from exiting.
        # The worker thread runs independently in the background.
        rospy.spin()
        rospy.loginfo("Navigation node shutting down run loop.")

    def _shutdown(self):
        rospy.loginfo("Shutting down Navigation 2D Node...")
        # Optional: Signal the worker thread to stop cleanly if needed,
        # e.g., by putting a special value (None) in the queue.
        # For now, rely on rospy.is_shutdown() check in the worker loop.

        if self._worker_thread is not None and self._worker_thread.is_alive():
            rospy.loginfo("Waiting briefly for worker thread to complete...")
            self._worker_thread.join(timeout=2.0) # Wait max 2 seconds
            if self._worker_thread.is_alive():
                rospy.loginfo("Worker thread did not finish cleanly after 2s.")
        rospy.loginfo("Shutdown complete.")

if __name__ == "__main__":
    try:
        rospy.init_node("navigation_2d_node")
        node = Navigation2dNode(rospy.get_name())
        node.run() # This will now call rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in main: {e}", exc_info=True)