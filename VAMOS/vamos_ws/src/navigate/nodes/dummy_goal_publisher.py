#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

def publish_goal():
    rospy.init_node('goal_publisher')
    
    # Get parameters from config or use defaults
    goal_topic = rospy.get_param('~goal_topic', '/spot/goal')  # Default to spot goal topic
    global_frame = rospy.get_param('~global_frame', 'map')  # Default to map frame
    
    pub = rospy.Publisher(goal_topic, PoseStamped, queue_size=10)
    rate = rospy.Rate(10)
    
    rospy.loginfo(f"Publishing goals to topic: {goal_topic} in frame: {global_frame}")
    
    while not rospy.is_shutdown():
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()  # Updated timestamp each time
        msg.header.frame_id = global_frame
        msg.pose.position.x = 5.0 #8.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0 #0.18
        msg.pose.orientation.w = 1.0
        
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_goal()
    except rospy.ROSInterruptException:
        pass
