# ROS2 Detection Package

ROS2 detector node for YOLO/Mask R-CNN object detection with RealSense depth fusion.


## Service/Topic API

Control is service-based:

- `/detector/init` (`ros2_detection_interfaces/srv/Init`): loads model and initializes camera.
- `/detector/start` (`ros2_detection_interfaces/srv/Start`): starts continuous detection.
- `/detector/stop` (`ros2_detection_interfaces/srv/Stop`): pauses detection loop.
- `/detector/release` (`ros2_detection_interfaces/srv/Release`): releases detector resources.

Monitoring is topic-based:

- `/detector/available_models`: periodic JSON array of available `model_path` strings.
- `/detector/status`: runtime status and error messages.
- `/detector/model_info`: model metadata after init.
- `/detector/results`: continuous detection payload while running.

Message types:

- Services: custom ROS2 service types in `ros2_detection_interfaces/srv/*`
- Topics: `std_msgs/String` with JSON payload strings

RGBD input is configured with the `rgbd_topics` node parameter. The default is
`["/sensors/realsense_rear/rgbd"]`. Multiple RGBD topics can be subscribed at
the same time; the detector uses the newest received frame.

Example multi-camera override:

```bash
ros2 run ros2_detection detector_node --ros-args -p rgbd_topics:="['/sensors/realsense_rear/rgbd','/sensors/realsense_front/rgbd']"
```

## Required Order

Use the interfaces in this order:

1. Start node (`ros2 launch ...`)
2. Call `/detector/init`
3. Call `/detector/start`
4. Read `/detector/results` (and optionally `/detector/status`, `/detector/model_info`)
5. Call `/detector/stop`
6. Call `/detector/release`


## Service Call Example (Manual)

Example `/detector/init` request:

```bash
ros2 service call /detector/init ros2_detection_interfaces/srv/Init "{model_type: yolo, model_path: model/tennisball_600_seg_yolo11_v02.pt, classes: ["Tennisball"], confidence: 0.5, use_decimation: false, use_spatial: false, use_temporal: true, use_hole_filling: true, use_mask_filter: true, color_resolution_width: 640, color_resolution_height: 480, use_realsense_ros_wrapper: True, rcnn_class_names: []}"
ros2 service call /detector/start ros2_detection_interfaces/srv/Start "{}"
ros2 service call /detector/stop ros2_detection_interfaces/srv/Stop "{}"
ros2 service call /detector/release ros2_detection_interfaces/srv/Release "{}"
```

Important `Init` fields:

- `use_realsense_ros_wrapper`: bool
- `model_path`: path to weights file
- `classes`: class filter list, e.g. `[Tennisball]`
- `confidence`: threshold (e.g. `0.5`)

Important node parameters:

- `rgbd_topics`: list of `realsense2_camera_msgs/msg/RGBD` input topics

You can also monitor outputs directly:

```bash
ros2 topic echo /detector/available_models
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

