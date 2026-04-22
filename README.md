### Set up the Repository
Setup Docker --> Explain here
Clone the repository
```bash
git clone https://github.com/astark146/crv_fieldrobotevent.git
cd crv_fieldrobotevent
```

Create and activate a Python virtual environment (Optional)
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install all required dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ROS2 Detector Node

A ROS2 detector node is available in `ros2_detection/ros2_detection/detector_node.py`.

It wraps the package-local detection implementation and supports runtime model selection (`yolo` or `rcnn`) with service-based control and JSON-based status/result output.

## Current Layout

The repository is split into two clear layers:

- `ros2_detection/` contains the active ROS2 package and the real runtime code.
- `legacy/` contains compatibility entrypoints for older scripts that imported the project before ROS2.
- The workspace root is now kept clean and only holds project-level files like the Docker and package docs.

## Model Files

Place new weights in the workspace-level `model/` folder when possible.

available YOLO models: 
- `model/tennisball_600_seg_yolo11_v02.pt`
- `model/yolo11_jute_stripe_yellow_paper-seg.pt`
- `model/yolo11n-seg.pt`

If you set `model_path` to just the filename, the node will search the workspace root and `model/` automatically.
The Docker image also copies `model/` into `/root/ros2_ws/model`.

Control services (order):
- `/detector/init` -> `/detector/start` -> `/detector/stop` -> `/detector/release`

Monitoring topics:
- `/detector/model_info` (`std_msgs/String`, JSON)
- `/detector/results` (`std_msgs/String`, JSON)
- `/detector/status` (`std_msgs/String`, JSON)

Run the node:
```bash
ros2 run ros2_detection detector_node
```

Build and source first:
```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_detection_interfaces ros2_detection
source install/setup.bash
```

Recommended control client:
```bash
ros2 run ros2_detection detector_client run --model-path model/yolo11_jute_stripe_yellow_paper-seg.pt --confidence 0.5 --duration 15 --use-realsense-ros-wrapper True
```

When using with realsense ros wrapper:
start the realsene laun file with the follwoing parameters:
```bash
ros2 launch realsense2_camera rs_launch.py \ 
enable_rgbd:=true \ 
enable_sync:=true \
align_depth.enable:=true \
enable_color:=true \
enable_depth:=true \
color_module.profile:=640x480x30
```

For detailed usage, see:
- `ros2_detection/README.md`
- `ros2_detection/ros2_detection/detector_client.py`

