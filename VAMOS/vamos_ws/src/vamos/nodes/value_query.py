#!/usr/bin/env python
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray

def traversability_estimator(origin, goals, trav_map, resolution, n=20):
    """Estimate traversability along a path defined by a list of goals."""
    # Get x, y components of the goals
    goals = goals[:, :, :2]  # ignore z
    # Convert meters → cell offsets
    
    goals[:,:,0] /= resolution      # +x → +columns
    goals[:,:,1] /= -resolution      # +y → -rows (image row 0 is top)

    starts = np.zeros_like(goals)
    starts[:,1:,:] = goals[:,:-1,:]

    num_goals = goals.shape[0]

    hypot = np.linalg.norm(goals - starts, axis=2) + 1e-6
    samples = np.linspace(0, 1, n)
    rads = hypot[:, :, np.newaxis] * samples[np.newaxis, np.newaxis, :]
    x_scale = (goals[:, :, 0]-starts[:, :, 0]) / hypot # N,T,2
    y_scale = (goals[:, :, 1]-starts[:, :, 1]) / hypot
    pts = origin + starts[:,:,np.newaxis,:] + rads[:,:,:, np.newaxis] * np.stack((x_scale, y_scale), axis=-1)[:,:,np.newaxis,:]
    pts = np.round(pts).astype(int)

    # Sample traversability, clipping out-of-bounds
    valid_mask = (0 <= pts[..., 1]) & (pts[..., 1] < trav_map.shape[0]) & (0 <= pts[..., 0]) & (pts[..., 0] < trav_map.shape[1])
    valid_pts = np.where(valid_mask[:,:,:,np.newaxis],
        pts,
        origin
    )
    valid_pts = valid_pts.astype(np.int32)
    values = trav_map[valid_pts[..., 1], valid_pts[..., 0]].max(axis=(1,2)).astype(float).tolist()
    
    return values

class TraversabilityEstimator(object):
    def __init__(self):
        # parameters
        self.image_topic = rospy.get_param('~image_topic', '/vamos/value_map_image')
        self.clicked_point_topic = rospy.get_param('~clicked_point_topic', '/clicked_point')
        self.resolution = rospy.get_param('~resolution', 0.25)  # meters/cell

        self.bridge = CvBridge()
        self.trav_map = None

        # subscribers & publisher
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.Subscriber(self.clicked_point_topic, PointStamped, self.goal_cb, queue_size=1)
        self.pub = rospy.Publisher(
            '/path_traversability', Float32MultiArray, queue_size=1
        )

        rospy.loginfo("TraversabilityEstimator ready: listening on %s and %s",
                      self.image_topic, self.clicked_point_topic)

    def image_cb(self, msg):
        """Convert incoming Image to a numpy array of float32 traversability."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.trav_map = np.array(cv_img, copy=False)
            # self.trav_map = np.flipud(trav_map)
            self.h, self.w = self.trav_map.shape
            # assume robot is at the center cell
            self.center = np.array([0.0 ,self.h // 2])
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)

    def goal_cb(self, msg):
        """When a goal is clicked, compute & publish the traversability along the path."""
        if self.trav_map is None:
            rospy.logwarn("No traversability map yet; can't compute path.")
            return

        # Extract goal in map frame (assumed same frame as the image)
        x = msg.point.x
        y = msg.point.y

        # Convert meters → cell offsets
        values = traversability_estimator(self.center, np.array([[[x, y],[x+0.5,y+0.5],[x+1.0,y+1.0]]]), self.trav_map, self.resolution)
        if not values:
            rospy.logwarn("Clicked goal is completely outside the map bounds.")
            return

        # Publish as Float32MultiArray
        arr = Float32MultiArray()
        arr.data = values
        self.pub.publish(arr)
        rospy.loginfo("Published %d traversability samples.", len(values))

if __name__ == '__main__':
    rospy.init_node('traversability_estimator')
    # TraversabilityEstimator()
    # rospy.spin()
    elevation_map = np.zeros((21, 21), dtype=np.float32)
    elevation_map[:,10] = 1.0
    start = np.array([0, 10])
    goals = np.array([[[0, 0], [-1,-1], [-1.5,0]], [[0.5, 0.5], [1, 1],[2,-1]]])
    resolution = 0.25
    print(traversability_estimator(start, goals, elevation_map, resolution))