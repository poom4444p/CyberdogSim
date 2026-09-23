# Navigate Package

This package provides navigation capabilities for robots using VLM (Vision-Language Model) trajectory prediction.

## Overview

The navigate package includes:
- Navigation node with VLM-based trajectory prediction
- Configurable goal publishers for testing
- Support for different robot configurations (Spot, Hound, Custom)
- Debug capabilities for topic monitoring

## Configuration Files

### Robot-Specific Configurations

- **`navigate_spot.yaml`**: Configuration for Spot robot with ZED2i camera
- **`navigate_hound.yaml`**: Configuration for Hound robot with standard camera
- **`navigate_custom.yaml`**: Custom configuration template for testing

### Custom Configuration

The custom configuration (`navigate_custom.yaml`) is designed for testing and development:

```yaml
image_topic: ""                    # Set your image topic
depth_topic: ""                    # Set your depth topic  
depth_camera_info_topic: ""        # Set your camera info topic
goal_topic: ""                     # Set your goal topic
robot_frame: ""                    # Set your robot frame
predictor_type: "vlm"              # Different types only for testing
time_threshold: 60                 # Time in seconds before the VLM times out and replans
debug_subscribers: false           # Set true to debug topic subscribers
```

## Launch Files

### Standard Launch Files
- **`image_navigate_spot.launch`**: Launch navigate node for Spot robot
- **`image_navigate_hound.launch`**: Launch navigate node for Hound robot

### Custom Launch File
- **`image_navigate_custom.launch`**: Launch navigate node with custom configuration

## Usage

### Running the Navigation Node

#### For Spot Robot:
```bash
roslaunch navigate image_navigate_spot.launch
```

#### For Hound Robot:
```bash
roslaunch navigate image_navigate_hound.launch
```

#### For Custom Configuration:
```bash
roslaunch navigate image_navigate_custom.launch
```

### Running the Dummy Goal Publisher

The dummy goal publisher can be used to test the navigation system:

#### Basic Usage (uses default Spot configuration):
```bash
rosrun navigate dummy_goal_publisher.py
```

#### With Custom Goal Topic:
```bash
rosrun navigate dummy_goal_publisher.py _goal_topic:=/custom/goal
```

#### With Custom Global Frame:
```bash
rosrun navigate dummy_goal_publisher.py _global_frame:=odom
```

#### For Hound Robot:
```bash
rosrun navigate dummy_goal_publisher.py _goal_topic:=/hound/goal
```

### Complete Testing Setup

To test the navigation system with custom configuration:

1. **Configure your topics** in `navigate_custom.yaml`:
   ```yaml
   image_topic: "/your_camera/image_raw"
   depth_topic: "/your_camera/depth/image_raw"
   depth_camera_info_topic: "/your_camera/depth/camera_info"
   goal_topic: "/custom/goal"
   robot_frame: "base_link"
   ```

2. **Launch the navigation node**:
   ```bash
   roslaunch navigate image_navigate_custom.launch
   ```

3. **Launch the dummy goal publisher**:
   ```bash
   rosrun navigate dummy_goal_publisher.py _goal_topic:=/custom/goal
   ```

## Parameters

### Navigation Node Parameters

- **`image_topic`**: ROS topic for RGB images
- **`depth_topic`**: ROS topic for depth images  
- **`depth_camera_info_topic`**: ROS topic for camera calibration info
- **`goal_topic`**: ROS topic for navigation goals
- **`robot_frame`**: TF frame of the robot base
- **`predictor_type`**: Type of trajectory predictor ("vlm", "dummy", "base")
- **`time_threshold`**: Timeout for VLM processing (seconds)
- **`debug_subscribers`**: Enable debug logging for topic subscribers

### Dummy Goal Publisher Parameters

- **`goal_topic`**: Topic to publish goals to (default: `/spot/goal`)
- **`global_frame`**: Frame ID for published goals (default: `map`)

## Debugging

### Enable Debug Subscribers

Set `debug_subscribers: true` in your configuration file to enable detailed logging of incoming messages:

```yaml
debug_subscribers: true
```

### Monitor Topics

You can monitor the navigation system using standard ROS tools:

```bash
# Monitor goal messages
rostopic echo /custom/goal

# Monitor image topics
rostopic echo /your_camera/image_raw

# Check TF frames
rosrun tf tf_monitor
```

## Dependencies

- ROS (tested with ROS Noetic)
- Python 3
- NumPy
- OpenCV (cv_bridge)
- message_filters
- tf2_ros
- trajectory_prediction package
- vamos package

## Troubleshooting

### Common Issues

1. **No CameraInfo received**: Ensure your camera_info topic is publishing
2. **TF lookup failed**: Check that TF frames are available and properly configured
3. **Queue full errors**: The inference queue has a size limit; reduce processing time or increase queue size
4. **Empty topics**: Verify that all required topics are publishing data

### Debug Steps

1. Check topic availability:
   ```bash
   rostopic list
   rostopic echo /your_topic
   ```

2. Verify TF frames:
   ```bash
   rosrun tf tf_echo base_link map
   ```

3. Enable debug subscribers in configuration
4. Check ROS logs for error messages
