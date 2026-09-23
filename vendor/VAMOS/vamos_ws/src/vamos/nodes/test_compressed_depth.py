#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
from cv_bridge import CvBridge

class DepthImageSaver:
    def __init__(self):
        rospy.init_node('depth_image_saver_simple', anonymous=True)

        self.save_dir = rospy.get_param('~save_dir', '/tmp/depth_images_raw')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/zed2i/depth_registered_raw', Image, self.callback, queue_size=1)

        self.counter = 0
        rospy.loginfo("🟢 Depth Image Saver Node Started (simple raw subscriber)")
        rospy.spin()

    def callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            rospy.loginfo(f"Depth image stats: min={np.nanmin(depth_image)}, max={np.nanmax(depth_image)}")

            # Convert to uint16 (millimeters) for saving
            img_to_save = np.clip(depth_image * 1000, 0, 65535).astype(np.uint16)
            save_path = os.path.join(self.save_dir, f"depth_{self.counter:06d}.png")

            if cv2.imwrite(save_path, img_to_save):
                rospy.loginfo(f"✅ Saved {save_path}")
            else:
                rospy.logerr(f"❌ Failed saving {save_path}")

            self.counter += 1

        except Exception as e:
            rospy.logerr(f"Exception in callback: {e}")

if __name__ == '__main__':
    DepthImageSaver()
