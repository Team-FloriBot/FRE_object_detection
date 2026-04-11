# ROS2 Detection Package

ROS2 detector node for YOLO/Mask R-CNN object detection with RealSense depth fusion.

## Setup

1. **Build the package:**
   ```bash
   cd ros2_detection
   colcon build --packages-select ros2_detection
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

### Subscriptions

#### `/detector/config` (std_msgs/String with JSON)
Initialize, start, stop, or release detector.

**Actions:**
- `"init"` - Initialize detector with config
- `"start"` - Start detection loop
- `"stop"` - Stop detection (keep detector loaded)
- `"release"` - Clean up and release detector

**Example:**
```json
{
  "action": "init",
  "model_type": "yolo",
  "model_path": "tennisball_600_seg_yolo11_v02.pt",
  "classes": ["Tennisball"],
  "confidence": 0.5,
  "filters": {
    "use_decimation": false,
    "use_spatial": false,
    "use_temporal": true,
    "use_hole_filling": true,
    "use_mask_filter": true
  },
  "camera": {
    "color_resolution": [640, 480],
    "fps": 30
  }
}
```

#### `/detector/request` (std_msgs/String with JSON)
Trigger detection on current frame with optional class/confidence override.

**Fields:**
- `classes` (list): Classes to detect (e.g., `["Tennisball"]`)
- `confidence` (float): Confidence threshold (overrides config)
- `max_results` (int): Limit number of results, 0 = unlimited

**Example:**
```json
{
  "classes": ["Tennisball"],
  "confidence": 0.45,
  "max_results": 5
}
```

### Publications

#### `/detector/model_info` (std_msgs/String with JSON)
Published after successful detector initialization.

**Fields:**
- `model_type` - Type of model ("yolo" or "rcnn")
- `model_path` - Path to model weights
- `selected_classes` - Classes configured for detection
- `available_classes` - All classes available in model
- `conf` - Current confidence threshold

#### `/detector/results` (std_msgs/String with JSON)
Published after each detection request.

**Fields:**
- `ok` - Success flag (true/false)
- `searched_classes` - Classes that were searched
- `model` - Model info object
- `num_detections` - Number of detections found
- `detections` - Array of detection objects:
  - `class` - Class name
  - `confidence` - Detection confidence (if available)
  - `center_xy` - [x, y] pixel coordinates
  - `median_xyz` - [x, y, z] 3D median (if fusion available)
  - `balliness` - Sphere quality score (0-100)
  - `radius` - Estimated sphere radius in meters

**Example Response:**
```json
{
  "ok": true,
  "searched_classes": ["Tennisball"],
  "model": {
    "model_type": "yolo",
    "model_path": "tennisball_600_seg_yolo11_v02.pt",
    "selected_classes": ["Tennisball"],
    "available_classes": ["Tennisball"],
    "conf": 0.5
  },
  "num_detections": 2,
  "detections": [
    {
      "class": "Tennisball",
      "confidence": 0.87,
      "center_xy": [320, 240],
      "median_xyz": [0.15, 0.05, 0.8],
      "balliness": 92,
      "radius": 0.033
    }
  ]
}
```

#### `/detector/status` (std_msgs/String with JSON)
Status and error messages.

**Fields:**
- `level` - "ready", "ok", or "error"
- `message` - Status/error message

---

## Examples

### Shell Example: Initialize + Detect

**Terminal 1: Start the node**
```bash
ros2 run ros2_detection detector_node
```

**Terminal 2: Send initialization config**
```bash
ros2 topic pub /detector/config std_msgs/String \
  -1 "{data: '{\"action\":\"init\",\"model_type\":\"yolo\",\"model_path\":\"tennisball_600_seg_yolo11_v02.pt\",\"classes\":[\"Tennisball\"],\"confidence\":0.5,\"filters\":{\"use_decimation\":false,\"use_spatial\":false,\"use_temporal\":true,\"use_hole_filling\":true,\"use_mask_filter\":true},\"camera\":{\"color_resolution\":[640,480],\"fps\":30}}'}"
```

**Terminal 3: Send detection request**
```bash
ros2 topic pub /detector/request std_msgs/String \
  -1 "{data: '{\"classes\":[\"Tennisball\"],\"confidence\":0.45,\"max_results\":5}'}"
```

**Terminal 4: Monitor results (in another shell)**
```bash
ros2 topic echo /detector/results
ros2 topic echo /detector/status
```

### Python Example: Client

```python
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DetectorClient(Node):
    def __init__(self):
        super().__init__("detector_client")
        self.config_pub = self.create_publisher(String, "/detector/config", 10)
        self.request_pub = self.create_publisher(String, "/detector/request", 10)
        
        self.status_sub = self.create_subscription(
            String, "/detector/status", self._on_status, 10
        )
        self.results_sub = self.create_subscription(
            String, "/detector/results", self._on_results, 10
        )
    
    def init_detector(self):
        config = {
            "action": "init",
            "model_type": "yolo",
            "model_path": "tennisball_600_seg_yolo11_v02.pt",
            "classes": ["Tennisball"],
            "confidence": 0.5,
            "filters": {
                "use_temporal": True,
                "use_hole_filling": True,
                "use_mask_filter": True
            },
            "camera": {"color_resolution": [640, 480], "fps": 30}
        }
        msg = String(data=json.dumps(config))
        self.config_pub.publish(msg)
        self.get_logger().info("Sent init config")
    
    def request_detection(self):
        req = {
            "classes": ["Tennisball"],
            "confidence": 0.45,
            "max_results": 5
        }
        msg = String(data=json.dumps(req))
        self.request_pub.publish(msg)
        self.get_logger().info("Sent detection request")
    
    def _on_status(self, msg: String):
        data = json.loads(msg.data)
        self.get_logger().info(f"Status [{data['level']}]: {data['message']}")
    
    def _on_results(self, msg: String):
        data = json.loads(msg.data)
        self.get_logger().info(f"Detections: {data['num_detections']}")
        for det in data["detections"]:
            self.get_logger().info(f"  - {det['class']}: conf={det['confidence']}")

def main():
    rclpy.init()
    client = DetectorClient()
    client.init_detector()
    # Wait a bit for initialization
    client.create_timer(2.0, lambda: client.request_detection())
    rclpy.spin(client)

if __name__ == "__main__":
    main()
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

Terminal 3 (one-shot init publish):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub -1 /detector/config std_msgs/msg/String "{data: '$(tr -d '\n' < ros2_detection/example_config.json)'}"
```

Terminal 4 (model info):
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /detector/model_info
```
