# ROS2 Detection Package

ROS2 detector node for YOLO/Mask R-CNN object detection with RealSense depth fusion.

## Quickstart

From workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_detection_interfaces ros2_detection
source install/setup.bash
```

Start detector node:

```bash
ros2 launch ros2_detection detector.launch.py
```

## Interface Overview

Control is service-based:

- `/detector/init` (`ros2_detection_interfaces/srv/Init`): loads model and initializes camera.
- `/detector/start` (`ros2_detection_interfaces/srv/Start`): starts continuous detection.
- `/detector/stop` (`ros2_detection_interfaces/srv/Stop`): pauses detection loop.
- `/detector/release` (`ros2_detection_interfaces/srv/Release`): releases detector resources.

Monitoring is topic-based:

- `/detector/status`: runtime status and error messages.
- `/detector/model_info`: model metadata after init.
- `/detector/results`: continuous detection payload while running.

## Recommended Usage

Use the client helper for normal operation (init/start/listen/stop/release):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros2_detection detector_client run --duration 15
```

Single actions:

```bash
ros2 run ros2_detection detector_client init
ros2 run ros2_detection detector_client start
ros2 run ros2_detection detector_client stop
ros2 run ros2_detection detector_client release
```

You can also monitor outputs directly:

```bash
ros2 topic echo /detector/status
ros2 topic echo /detector/model_info
ros2 topic echo /detector/results
```

The most up-to-date usage flow is implemented in `ros2_detection/ros2_detection/detector_client.py`.

## Notes

- Model paths are resolved relative to the current working directory.
- A connected and configured RealSense camera is required.
