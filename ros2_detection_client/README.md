# ROS2 Detection Package

ROS2 detector node for YOLO/Mask R-CNN object detection with RealSense depth fusion.


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


## Bash Start With Parameters

If you want to start the detector client directly from Bash, use the client node:

```bash
ros2 run ros2_detection_client detector_client run \
	--model-path model/yolo26n_jute_stripe_yellow_paper_02-seg.pt \
	--confidence 0.5 \
	--duration 15 \
	--use-realsense-ros-wrapper
```

If you want to call the detector services manually in Bash, the order is:

```bash
ros2 service call /detector/init ros2_detection_interfaces/srv/Init '{
	model_type: yolo,
	model_path: model/yolo26n_jute_stripe_yellow_paper_02-seg.pt,
	classes: [],
	confidence: 0.5,
	use_decimation: false,
	use_spatial: false,
	use_temporal: true,
	use_hole_filling: true,
	use_mask_filter: true,
	use_realsense_ros_wrapper: true,
	color_resolution_width: 640,
	color_resolution_height: 480,
	fps: 30,
	rcnn_class_names: []
}'

ros2 service call /detector/start ros2_detection_interfaces/srv/Start '{}'
ros2 service call /detector/stop ros2_detection_interfaces/srv/Stop '{}'
ros2 service call /detector/release ros2_detection_interfaces/srv/Release '{}'
```

The important `Init` fields are:

- `model_type`
- `model_path`
- `classes`
- `confidence`
- `use_realsense_ros_wrapper`


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

