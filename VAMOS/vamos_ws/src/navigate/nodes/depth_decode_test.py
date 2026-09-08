#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os

OUTPUT_FILENAME = "depth_test_16uc1.png" # New name for clarity
DECODED_FILENAME = "depth_decoded_16uc1.png" # New name
RECEIVED = False

def callback(msg):
    global RECEIVED
    if RECEIVED:
        return

    rospy.loginfo("Received first compressed depth message.")
    rospy.loginfo(f"Message format: {msg.format}") # Log format
    RECEIVED = True

    # 1. Save the raw compressed data
    try:
        with open(OUTPUT_FILENAME, 'wb') as f:
            f.write(msg.data)
        rospy.loginfo(f"Saved compressed data to {OUTPUT_FILENAME}")
    except Exception as e:
        rospy.logerr(f"Failed to save raw data: {e}")
        rospy.signal_shutdown("Error saving data")
        return

    # 2. Try decoding directly using cv2.imdecode
    try:
        rospy.loginfo(f"Attempting cv2.imdecode on received data...")
        # Convert bytes to numpy array first
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Decode using cv2.IMREAD_UNCHANGED to preserve original depth type (should be uint16)
        decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if decoded_image is None:
            rospy.logerr("cv2.imdecode returned None! Cannot decode the PNG data.")
        else:
            rospy.loginfo(f"cv2.imdecode SUCCESS!")
            rospy.loginfo(f"  - Decoded shape: {decoded_image.shape}")
            rospy.loginfo(f"  - Decoded dtype: {decoded_image.dtype}") # <<< Should be uint16 now

            # Try saving the decoded image (should work for uint16)
            try:
                cv2.imwrite(DECODED_FILENAME, decoded_image)
                rospy.loginfo(f"Saved visually checkable decoded image to {DECODED_FILENAME}")
            except Exception as e:
                 rospy.logwarn(f"Could not save decoded image: {e}")

    except Exception as e:
        rospy.logerr(f"Error during cv2.imdecode: {e}")
        import traceback
        traceback.print_exc() # Print full traceback

    rospy.signal_shutdown("Finished test.")


if __name__ == "__main__":
    rospy.init_node("depth_decode_test", anonymous=True)
    # Make sure this topic matches your throttled topic
    topic = "/zed2i/zed_node/depth/depth_registered/compressedDepth_throttle"
    rospy.loginfo(f"Subscribing to {topic}")
    sub = rospy.Subscriber(topic, CompressedImage, callback)
    rospy.spin()