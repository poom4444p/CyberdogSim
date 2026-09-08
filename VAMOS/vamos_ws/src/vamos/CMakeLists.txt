cmake_minimum_required(VERSION 3.0.2)
project(vamos)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
catkin_python_setup()

catkin_package()

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Unsure if this code below is necessary
include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  # nodes/controller_node.py
  # nodes/mock_velocity_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)