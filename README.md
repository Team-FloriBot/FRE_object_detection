### Set up the Repository

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

A ROS2 topic-driven node is available in `ros2_detection/ros2_detection/detector_node.py`.

It wraps [detection_model_selection.py](detection_model_selection.py) and supports:
- Runtime model selection (`yolo` or `rcnn`)
- Runtime model path configuration
- Runtime class selection and confidence override per request
- Detection result publishing as JSON

Subscriptions:
- `/detector/config` (`std_msgs/String`, JSON)
- `/detector/request` (`std_msgs/String`, JSON)

Publications:
- `/detector/model_info` (`std_msgs/String`, JSON)
- `/detector/results` (`std_msgs/String`, JSON)
- `/detector/status` (`std_msgs/String`, JSON)

Run the node:
```bash
ros2 run ros2_detection detector_node
```

Build and source first:
```bash
colcon build --packages-select ros2_detection
source install/setup.bash
```

Example config message (`/detector/config`):
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

Example request message (`/detector/request`):
```json
{
  "classes": ["Tennisball"],
  "confidence": 0.45,
  "max_results": 5
}
```

Result payload (`/detector/results`) includes:
- `searched_classes`
- `model` (type/path/classes/conf)
- `detections[]` with `class`, `confidence`, `center_xy`, and if available `median_xyz`, `balliness`, `radius`

