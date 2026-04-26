### Set up

Clone the repository
```bash
git clone https://github.com/astark146/crv_fieldrobotevent.git
```

Build the docker container, takes some time
```bash
docker compose build
```

You need the ros2_detection_interfaces package in your client workspace. So copy it, when your client is elsewhere.


### Overview

The ROS2 detector node wraps the package-local detection implementation. The package can use a yolo model to detect and localize objects. 
The detector node control is service-based and publishes the camera results. See the README-file and detector_client node to get further informations about how to use the detector_node.
detector_client is an example implementation which shows the basic usage with the following services and topics.

Control services (order):
- `/detector/init` -> `/detector/start` -> `/detector/stop` -> `/detector/release`

Monitoring topics:
- `/detector/model_info` (`std_msgs/String`, JSON) 
- `/detector/results` (`std_msgs/String`, JSON)
- `/detector/status` (`std_msgs/String`, JSON)


## Model Files

To use your own YOLO model, add it to the workspace-level `model/` folder.

currently available YOLO models: 
- `model/tennisball_600_seg_yolo11_v02.pt`
- `model/yolo11_jute_stripe_yellow_paper-seg.pt`
- `model/yolo11_jute_stripe_yellow_paper_02-seg.pt`
- `model/yolo11n-seg.pt`


### Use the detector

Start the docker container to launch the detector_node.
```bash
docker compose up -d
```

Start the example detector_client in this workspace:
```bash
docker compose exec detecor_client -it bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros2_detection_client detector_client run --model-path model/yolo11_jute_stripe_yellow_paper-seg.pt --confidence 0.5 --duration 15 --use-realsense-ros-wrapper
```

When using with RealSense ROS wrapper:
start the RealSene launch file with the follwoing parameters:
```bash
ros2 launch realsense2_camera rs_launch.py \ 
enable_rgbd:=true \ 
enable_sync:=true \
align_depth.enable:=true \
enable_color:=true \
enable_depth:=true \
color_module.profile:=640x480x30
```

When using the RealSense camera on a Windows-system (either with the RealSense ROS wrapper, or directly with our detector_node) you need to attach tha camera to wsl with usbipd.


### For detailed usage, see:

- `ros2_detection/README.md`
- `ros2_detection_client/ros2_detection_client/detector_client.py`

