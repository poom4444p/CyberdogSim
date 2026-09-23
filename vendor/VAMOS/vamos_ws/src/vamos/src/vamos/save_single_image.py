#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class SingleShotSaver:
    def __init__(self):
        # topic and filename as ROS params (with defaults)
        topic     = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color/raw")
        topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        filename  = rospy.get_param("~output_file", "/home/rll/taller_hound_gates/pic0.png")
        self.encoding  = rospy.get_param("~encoding",    "bgr8")

        self.bridge = CvBridge()
        self.filename = filename

        rospy.loginfo(f"[single_shot] subscribing to {topic}, will save as {filename}")
        rospy.Subscriber(topic, Image, self.cb, queue_size=1)

    def cb(self, msg):
        # convert and save first frame, then exit
        cv_img = self.bridge.imgmsg_to_cv2(msg, self.encoding)
        cv2.imwrite(self.filename, cv_img)
        rospy.loginfo(f"[single_shot] saved image to {self.filename}")
        rospy.signal_shutdown("done")

if __name__ == "__main__":
    rospy.init_node("single_shot_saver", anonymous=True)
    saver = SingleShotSaver()
    rospy.spin()
