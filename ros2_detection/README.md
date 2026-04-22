# ROS2 Detection Package

ROS2 detector node for YOLO/Mask R-CNN object detection with RealSense depth fusion.

## Build & Start

From workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_detection_interfaces ros2_detection
source install/setup.bash
```

Start detector node (Terminal 1):

```bash
ros2 launch ros2_detection detector.launch.py
```

## Model Setup

Recommended model location:

- `model/` at workspace root, e.g. `model/tennisball_600_seg_yolo11_v02.pt`

Also possible:

- absolute path
- relative path from current working directory

For YOLO, pass the `.pt` file via `/detector/init` or `detector_client --model-path`.

## Service/Topic API

Control is service-based:

- `/detector/init` (`ros2_detection_interfaces/srv/Init`): loads model and initializes camera.
- `/detector/start` (`ros2_detection_interfaces/srv/Start`): starts continuous detection.
- `/detector/stop` (`ros2_detection_interfaces/srv/Stop`): pauses detection loop.
- `/detector/release` (`ros2_detection_interfaces/srv/Release`): releases detector resources.

Monitoring is topic-based:

- `/detector/status`: runtime status and error messages.
- `/detector/model_info`: model metadata after init.
- `/detector/results`: continuous detection payload while running.

Message types:

- Services: custom ROS2 service types in `ros2_detection_interfaces/srv/*`
- Topics: `std_msgs/String` with JSON payload strings

## Required Order

Use the interfaces in this order:

1. Start node (`ros2 launch ...`)
2. Call `/detector/init`
3. Call `/detector/start`
4. Read `/detector/results` (and optionally `/detector/status`, `/detector/model_info`)
5. Call `/detector/stop`
6. Call `/detector/release`

## Fastest Way (Client)

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros2_detection detector_client run --model-path model/yolo11_jute_stripe_yellow_paper-seg.pt --confidence 0.5 --duration 15 --use-realsense-ros-wrapper True
```

Single actions are available if needed:

```bash
ros2 run ros2_detection detector_client init --model-path model/tennisball_600_seg_yolo11_v02.pt
ros2 run ros2_detection detector_client start
ros2 run ros2_detection detector_client stop
ros2 run ros2_detection detector_client release
```

## Service Call Example (Manual)

Example `/detector/init` request:

```bash
ros2 service call /detector/init ros2_detection_interfaces/srv/Init "{model_type: yolo, model_path: model/tennisball_600_seg_yolo11_v02.pt, classes: ["Tennisball"], confidence: 0.5, use_decimation: false, use_spatial: false, use_temporal: true, use_hole_filling: true, use_mask_filter: true, color_resolution_width: 640, color_resolution_height: 480, use_realsense_ros_wrapper: True, rcnn_class_names: []}"
ros2 service call /detector/start ros2_detection_interfaces/srv/Start "{}"
ros2 service call /detector/stop ros2_detection_interfaces/srv/Stop "{}"
ros2 service call /detector/release ros2_detection_interfaces/srv/Release "{}"
```

Important `Init` fields:

- `model_type`: `yolo` or `rcnn`
- `model_path`: path to weights file
- `classes`: class filter list, e.g. `[Tennisball]`
- `confidence`: threshold (e.g. `0.5`)
- `fps`, `color_resolution_width`, `color_resolution_height`: camera setup

You can also monitor outputs directly:

```bash
ros2 topic echo /detector/status
ros2 topic echo /detector/model_info
ros2 topic echo /detector/results
```

Example `/detector/results` payload (JSON string in `std_msgs/String`):

```json
{
	"num_detections": 1,
	"detections": [
		{
			"class": "Tennisball",
			"confidence": 0.91,
			"median_xyz": [0.12, -0.03, 0.78],
			"object_center": [0.11, -0.02, 0.79]
		}
	]
}
```

## Notes

- Model paths are resolved relative to the current working directory.
- A connected and configured RealSense camera is required.
