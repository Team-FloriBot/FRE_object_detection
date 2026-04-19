# ROS2 Detection Package

ROS2 detector node for YOLO/Mask R-CNN object detection with RealSense depth fusion.

## Setup

1. **Build the package:**
   ```bash
   cd /workspace
   colcon build --packages-select ros2_detection_interfaces ros2_detection
   source install/setup.bash
   ```

2. **Source the setup:**
   ```bash
   source install/setup.bash
   ```

## Running the Node

### Method 1: Direct Node Execution
```bash
ros2 run ros2_detection detector_node
```

### Method 2: Using Launch File
```bash
ros2 launch ros2_detection detector.launch.py
```

## Topic Interface
The node now uses **services for control** and **topics for monitoring/results**.

## Control Flow

Use the node in this order:

1. Start the node with `ros2 launch ros2_detection detector.launch.py`.
2. Call `/detector/init` to load model + initialize camera.
3. Optionally read `/detector/model_info` and `/detector/status`.
4. Call `/detector/start` to start continuous detection.
5. Read `/detector/results` continuously.
6. Call `/detector/stop` to pause loop.
7. Call `/detector/release` to free camera/model.

## Service API

| Service | Type | Purpose |
|---|---|---|
| `/detector/init` | `ros2_detection_interfaces/srv/Init` | Initialize model and camera |
| `/detector/start` | `ros2_detection_interfaces/srv/Start` | Start continuous detection loop |
| `/detector/stop` | `ros2_detection_interfaces/srv/Stop` | Stop loop, keep detector loaded |
| `/detector/release` | `ros2_detection_interfaces/srv/Release` | Stop + release detector resources |

## Topic API

| Topic | Direction | Purpose |
|---|---|---|
| `/detector/model_info` | subscribe | Model metadata after successful init |
| `/detector/status` | subscribe | Status and error messages |
| `/detector/results` | subscribe | Continuous detection output while started |

## Quick Smoke Test

Use these commands from the workspace root in separate terminals.

Terminal 1:
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ros2_detection detector.launch.py
```

Terminal 2 (status monitor):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /detector/status
```

Terminal 3 (results monitor):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /detector/results
```

Terminal 4 (init):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 service call /detector/init ros2_detection_interfaces/srv/Init "{model_type: yolo, model_path: models/tennisball_600_seg_yolo11_v02.pt, classes: [Tennisball], confidence: 0.5, use_decimation: false, use_spatial: false, use_temporal: true, use_hole_filling: true, use_mask_filter: true, color_resolution_width: 640, color_resolution_height: 480, fps: 30, rcnn_class_names: []}"
```

Terminal 4 (start):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 service call /detector/start ros2_detection_interfaces/srv/Start "{}"
```

Terminal 4 (stop/release):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 service call /detector/stop ros2_detection_interfaces/srv/Stop "{}"
ros2 service call /detector/release ros2_detection_interfaces/srv/Release "{}"
```

## Python Client Helper

The package now includes a helper client executable that calls services and prints
`/detector/status`, `/detector/model_info`, and `/detector/results` live.

Run full flow (`init -> start -> listen -> stop -> release`):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros2_detection detector_client run --duration 15
```

Run single actions:
```bash
ros2 run ros2_detection detector_client init
ros2 run ros2_detection detector_client start
ros2 run ros2_detection detector_client stop
ros2 run ros2_detection detector_client release
```

## Dependencies

Required Python packages (should be in parent project's `requirements.txt`):
- `torch`
- `torchvision`
- `ultralytics` (YOLO)
- `opencv-python`
- `pyrealsense2`
- `open3d`
- `scikit-learn`
- `numpy`

## Notes

- The detector uses the package-local implementation in `ros2_detection/detection_model_selection.py`
- Model paths are relative to the working directory where the node is launched
- The RealSense camera must be connected and properly configured
- For RCNN, ensure the weights file path is correct
