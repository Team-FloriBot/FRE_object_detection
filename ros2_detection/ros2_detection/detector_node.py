import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from ros2_detection_interfaces.srv import Init, Release, Start, Stop
from rclpy.clock import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD
from ros2_detection_interfaces.msg import Detection, DetectionArray
from geometry_msgs.msg import Point
import cv2

from .detection_model_selection import ObjDetection


class DetectionNode(Node):
    """
    ROS2 node that wraps ObjDetection with a continuous detection loop.
    
        Services:
            - /detector/init (ros2_detection_interfaces/Init)
            - /detector/start (ros2_detection_interfaces/Start)
            - /detector/stop (ros2_detection_interfaces/Stop)
            - /detector/release (ros2_detection_interfaces/Release)
    
    Topics (Publications):
            - /detector/available_models (std_msgs/String with JSON array of model_path strings)
      - /detector/model_info (std_msgs/String with JSON)
      - /detector/results (std_msgs/String with JSON)
      - /detector/status (std_msgs/String with JSON)
    """

    def __init__(self) -> None:
        super().__init__("detector_node")

        self.default_rgbd_topics = ["/sensors/realsense_rear/rgbd"]
        self.detector: Optional[ObjDetection] = None
        self.detector_running = False
        self.loop_timer = None
        self.available_models_timer = None
        self.detection_period_sec = 0.0
        self.current_config: Dict[str, Any] = {}

        # --- Node parameters (declared with defaults) ---
        # Model / detection
        self.declare_parameter("model_type", "yolo")
        self.declare_parameter("model_path", "")
        self.declare_parameter("classes", [])
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("max_detections", 50)

        # Filters
        self.declare_parameter("use_decimation", False)
        self.declare_parameter("use_spatial", True)
        self.declare_parameter("use_temporal", True)
        self.declare_parameter("use_hole_filling", False)
        self.declare_parameter("use_mask_filter", True)

        # Camera / input
        self.declare_parameter("use_realsense_ros_wrapper", False)
        self.declare_parameter("color_resolution_width", 640)
        self.declare_parameter("color_resolution_height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("depth_scale_override", 0.0)
        self.declare_parameter("rgbd_topics", self.default_rgbd_topics)

        # Topics / publishing
        self.declare_parameter("publish_annotated_image", False)
        self.declare_parameter("annotated_image_topic", "/detector/annotated_image")
        self.declare_parameter("results_topic", "/detector/results")
        self.declare_parameter("status_topic", "/detector/status")
        self.declare_parameter("available_models_topic", "/detector/available_models")
        self.declare_parameter("available_models_publish_period_sec", 10.0)

        # Read params into a dict for easy access
        self.node_params = {
            "model_type": self.get_parameter("model_type").value,
            "model_path": self.get_parameter("model_path").value,
            "classes": self.get_parameter("classes").value,
            "confidence": float(self.get_parameter("confidence").value),
            "max_detections": int(self.get_parameter("max_detections").value),
            "use_decimation": bool(self.get_parameter("use_decimation").value),
            "use_spatial": bool(self.get_parameter("use_spatial").value),
            "use_temporal": bool(self.get_parameter("use_temporal").value),
            "use_hole_filling": bool(self.get_parameter("use_hole_filling").value),
            "use_mask_filter": bool(self.get_parameter("use_mask_filter").value),
            "use_realsense_ros_wrapper": bool(self.get_parameter("use_realsense_ros_wrapper").value),
            "color_resolution": [int(self.get_parameter("color_resolution_width").value), int(self.get_parameter("color_resolution_height").value)],
            "fps": int(self.get_parameter("fps").value),
            "depth_scale_override": float(self.get_parameter("depth_scale_override").value),
            "rgbd_topics": self._normalize_rgbd_topics(self.get_parameter("rgbd_topics").value),
            "publish_annotated_image": bool(self.get_parameter("publish_annotated_image").value),
            "annotated_image_topic": self.get_parameter("annotated_image_topic").value,
            "results_topic": self.get_parameter("results_topic").value,
            "status_topic": self.get_parameter("status_topic").value,
            "available_models_topic": self.get_parameter("available_models_topic").value,
            "available_models_publish_period_sec": float(self.get_parameter("available_models_publish_period_sec").value),
        }

        # Publishers for monitoring (use parameterized topic names)
        self.model_info_pub = self.create_publisher(String, "/detector/model_info", 10)
        self.results_pub = self.create_publisher(DetectionArray, self.node_params.get("results_topic", "/detector/results"), 10)
        self.status_pub = self.create_publisher(String, self.node_params.get("status_topic", "/detector/status"), 10)
        self.available_models_pub = self.create_publisher(String, self.node_params.get("available_models_topic", "/detector/available_models"), 10)
        if self.node_params.get("publish_annotated_image", True):
            self.annotated_img_pub = self.create_publisher(Image, self.node_params.get("annotated_image_topic", "/detector/annotated_image"), 10)
        else:
            self.annotated_img_pub = None

        publish_period_sec = float(self.node_params.get("available_models_publish_period_sec", 10.0))
        if publish_period_sec <= 0.0:
            publish_period_sec = 10.0
        self.available_models_timer = self.create_timer(publish_period_sec, self._publish_available_models)
  
        self.last_frame_id = "camera_color_optical_frame"

        # Services for control
        self.init_service = self.create_service(
            Init,
            "/detector/init",
            self._handle_init,
        )
        self.start_service = self.create_service(
            Start,
            "/detector/start",
            self._handle_start,
        )
        self.stop_service = self.create_service(
            Stop,
            "/detector/stop",
            self._handle_stop,
        )
        self.release_service = self.create_service(
            Release,
            "/detector/release",
            self._handle_release,
        )


        self.bridge = CvBridge()
        self.color_img = None
        self.depth_img = None
        self.depth_camera_info = None
        self.last_stamp = None
        self.last_rgbd_topic = None
        self._rgbd_receive_seq = 0
        self.latest_rgbd_by_topic: Dict[str, Dict[str, Any]] = {}
        self._last_waiting_frame_status_sec = 0.0

        self.subs = []
        for topic in self.node_params["rgbd_topics"]:
            self.subs.append(
                self.create_subscription(
                    RGBD,
                    topic,
                    lambda msg, topic=topic: self.realsense_callback(msg, topic),
                    10
                )
            )

        self._publish_available_models()
        self._publish_status("ready", "Detector node started. Waiting for /detector/init.")

    def _normalize_rgbd_topics(self, topics) -> List[str]:
        """Return a non-empty list of RGBD topic names."""
        if isinstance(topics, str):
            normalized_topics = [topics]
        elif isinstance(topics, (list, tuple)):
            normalized_topics = [str(topic) for topic in topics if isinstance(topic, str) and topic.strip()]
        else:
            normalized_topics = []

        if not normalized_topics:
            self.get_logger().warn(
                f"Invalid or empty rgbd_topics parameter. Falling back to {self.default_rgbd_topics}."
            )
            return list(self.default_rgbd_topics)

        return normalized_topics

    def _stamp_to_nanoseconds(self, stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def realsense_callback(self, msg, topic: str) -> None:
        color_img = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        stamp = msg.rgb.header.stamp
        self._rgbd_receive_seq += 1

        self.latest_rgbd_by_topic[topic] = {
            "color_img": color_img,
            "depth_img": depth_img,
            "depth_camera_info": getattr(msg, "depth_camera_info", None),
            "stamp": stamp,
            "stamp_ns": self._stamp_to_nanoseconds(stamp),
            "receive_seq": self._rgbd_receive_seq,
            "frame_id": msg.rgb.header.frame_id,
        }

    # Ensure color image is not modified by OpenCV operations
    def get_realsense_images(self):
        if not self.latest_rgbd_by_topic:
            return None, None, None, None

        topic, frame = max(
            self.latest_rgbd_by_topic.items(),
            key=lambda item: (item[1]["stamp_ns"], item[1]["receive_seq"]),
        )

        self.color_img = frame["color_img"]
        self.depth_img = frame["depth_img"]
        self.depth_camera_info = frame["depth_camera_info"]
        self.last_stamp = frame["stamp"]
        self.last_frame_id = frame["frame_id"]
        self.last_rgbd_topic = topic

        return self.color_img.copy(), self.depth_img.copy(), self.last_frame_id, self.last_stamp

    def _handle_init(self, request, response) -> None:
        """Initialize detector with given configuration."""
        try:
            # Build config dict from service request, falling back to node params when fields are empty
            req_classes = list(request.classes) if request.classes and len(request.classes) > 0 else list(self.node_params.get("classes", []))
            req_model_type = request.model_type if getattr(request, "model_type", None) else self.node_params.get("model_type")
            req_model_path = request.model_path if getattr(request, "model_path", None) else self.node_params.get("model_path")
            req_confidence = float(request.confidence) if getattr(request, "confidence", None) else float(self.node_params.get("confidence", 0.5))
            req_rcnn_names = list(request.rcnn_class_names) if getattr(request, "rcnn_class_names", None) and len(request.rcnn_class_names) > 0 else None

            camera_use_wrapper = bool(request.use_realsense_ros_wrapper) if hasattr(request, "use_realsense_ros_wrapper") else bool(self.node_params.get("use_realsense_ros_wrapper", True))
            color_res_w = int(request.color_resolution_width) if hasattr(request, "color_resolution_width") and request.color_resolution_width > 0 else int(self.node_params.get("color_resolution", [640,480])[0])
            color_res_h = int(request.color_resolution_height) if hasattr(request, "color_resolution_height") and request.color_resolution_height > 0 else int(self.node_params.get("color_resolution", [640,480])[1])
            fps_val = int(request.fps) if hasattr(request, "fps") and request.fps > 0 else int(self.node_params.get("fps", 30))

            config = {
                "model_type": req_model_type,
                "model_path": req_model_path,
                "classes": req_classes,
                "confidence": req_confidence,
                "rcnn_class_names": req_rcnn_names,
                "filters": {
                    "use_decimation": bool(request.use_decimation) if hasattr(request, "use_decimation") else bool(self.node_params.get("use_decimation", False)),
                    "use_spatial": bool(request.use_spatial) if hasattr(request, "use_spatial") else bool(self.node_params.get("use_spatial", True)),
                    "use_temporal": bool(request.use_temporal) if hasattr(request, "use_temporal") else bool(self.node_params.get("use_temporal", True)),
                    "use_hole_filling": bool(request.use_hole_filling) if hasattr(request, "use_hole_filling") else bool(self.node_params.get("use_hole_filling", False)),
                    "use_mask_filter": bool(request.use_mask_filter) if hasattr(request, "use_mask_filter") else bool(self.node_params.get("use_mask_filter", True)),
                },
                "camera": {
                    "use_realsense_ros_wrapper": camera_use_wrapper,
                    "color_resolution": [color_res_w, color_res_h],
                    "fps": fps_val,
                },
            }

            if not isinstance(config["classes"], list):
                response.success = False
                response.message = "Config field 'classes' must be a list."
                return response

            self._release_detector()

            # Initialize ObjDetection
            self.detector = ObjDetection(
                classes=config["classes"],
                model_type=config["model_type"],
                model_path=config["model_path"],
                rcnn_class_names=config["rcnn_class_names"],
                use_decimation=config["filters"]["use_decimation"],
                use_spatial=config["filters"]["use_spatial"],
                use_temporal=config["filters"]["use_temporal"],
                use_hole_filling=config["filters"]["use_hole_filling"],
                use_mask_filter=config["filters"]["use_mask_filter"],
                conf=config["confidence"]
            )

             
            
            # Initialize camera
            self.detector.initialize_realsense(
                color_resolution=tuple(config["camera"]["color_resolution"]),
                fps=config["camera"]["fps"],
                get_realsense_images = self.get_realsense_images if config["camera"]["use_realsense_ros_wrapper"] else None
            )

            # Allow node-level override of depth scale (useful for non-standard publishers)
            #depth_override = float(self.node_params.get("depth_scale_override", 0.0))
            #if depth_override and depth_override > 0.0:
            #    self.detector.depth_scale = depth_override
            #el
            if config["camera"]["use_realsense_ros_wrapper"] and self.detector.depth_scale is None:
                # RealSense ROS wrapper depth image is typically uint16 in millimeters.
                self.detector.depth_scale = 0.001

            self.current_config = config
            self.detector_running = False  # Start will be a separate call

            # Publish model info
            model_info = self.detector.get_model_info()
            self.model_info_pub.publish(String(data=json.dumps(model_info)))

            response.success = True
            response.message = "Detector initialized successfully."
            response.model_info = json.dumps(model_info)
            self._publish_status("ok", "Detector initialized.")
            return response

        except Exception as exc:
            self.detector = None
            response.success = False
            response.message = f"Failed to initialize detector: {exc}"
            self._publish_status("error", f"Initialization failed: {exc}")
            return response

    def _handle_start(self, request, response) -> None:
        """Start detection loop."""
        if self.detector is None:
            response.success = False
            response.message = "Detector not initialized. Call /detector/init first."
            return response

        if self.detector_running:
            response.success = False
            response.message = "Detector already running."
            return response

        # Use fixed period for now to avoid frequent timer destruction/creation
        fps = int(self.current_config.get("camera", {}).get("fps", 30))
        if fps <= 0:
            fps = 30
        self.detection_period_sec = 1.0 / fps  # seconds
        
        self.detector_running = True
        self.loop_timer = self.create_timer(self.detection_period_sec, self._detection_loop)

        response.success = True
        response.message = f"Detector started at {fps} FPS."
        self._publish_status("ok", f"Adaptive detection loop started at {fps} FPS.")
        return response

    def _handle_stop(self, request, response) -> None:
        """Stop continuous detection loop but keep detector loaded."""
        if self.loop_timer is not None:
            self.destroy_timer(self.loop_timer)
            self.loop_timer = None

        self.detector_running = False
        response.success = True
        response.message = "Detector stopped."
        self._publish_status("ok", "Detection loop stopped.")
        return response

    def _handle_release(self, request, response) -> None:
        """Release detector and cleanup."""
        self._release_detector()
        response.success = True
        response.message = "Detector released."
        self._publish_status("ready", "Detector released. Ready for new /detector/init.")
        return response

    def _detection_loop(self) -> None:
        """Run one detection cycle."""
        if self.detector is None or not self.detector_running:
            return

        try:
            color_image, depth_image, frame_id, timestamp = None, None, None, None
            
            if self.detector.get_realsense_images is not None:
                color_image, depth_image, frame_id, timestamp = self.detector.get_frame()

                # ROS wrapper can briefly return no images at startup.
                if color_image is None or depth_image is None:
                    now = time.monotonic()
                    if now - self._last_waiting_frame_status_sec > 2.0:
                        topics = ", ".join(self.node_params.get("rgbd_topics", self.default_rgbd_topics))
                        self._publish_status("ready", f"Waiting for RGBD frames on: {topics}.")
                        self._last_waiting_frame_status_sec = now
                    return

                # Depth from ROS wrapper is usually uint16 in mm, but some setups publish float meters.
                if self.detector.depth_scale is None:
                    if np.issubdtype(depth_image.dtype, np.floating):
                        self.detector.depth_scale = 1.0
                    else:
                        self.detector.depth_scale = 0.001

                self._update_detector_intrinsics_from_camera_info()
            else:
                frames = self.detector.get_frame()
                frame_id = self.last_frame_id
                timestamp = Clock().now().to_msg()
                aligned_frames = self.detector.align_frames(frames)
                color_image, depth_image = self.detector.depth_filter(aligned_frames)

            if color_image is None or depth_image is None:
                self._publish_status("error", "No valid color/depth frame available.")
                return

            if self.detector.fx == 0 or self.detector.fy == 0 or self.detector.depth_scale is None:
                self._publish_status("error", "Missing camera intrinsics or depth scale.")
                return
            
            
            objs_data, annotated_image = self.detector.detect_obj(color_image)
            

            fused_objs_data, _ = self.detector.fuse(color_image, depth_image, objs_data)

            detections = []
            msg = DetectionArray()

            msg.header.stamp = timestamp
            msg.header.frame_id = frame_id

            msg.ok = True
            msg.searched_classes = self.detector.classes

            msg.model = json.dumps(self.detector.get_model_info())

            for obj_id, data in fused_objs_data.items():
                det = Detection()

                det.id = str(obj_id)
                det.label = data.get("class", "")
                det.confidence = float(data.get("confidence", 0.0))

                mx, my, mz = data.get("median_xyz", [0,0,0])
                det.median_xyz = Point(x=mx, y=my, z=mz)

                ox, oy, oz = data.get("object_center", [0,0,0])
                det.object_center = Point(x=ox, y=oy, z=oz)

                msg.detections.append(det)

            self.results_pub.publish(msg)

            # Publish annotated image
            if annotated_image is not None:
                if self.annotated_img_pub is not None:
                    img_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.header.frame_id = frame_id if frame_id is not None else self.last_frame_id
                    self.annotated_img_pub.publish(img_msg)

        except Exception as exc:
            self._publish_status("error", f"Detection loop error: {exc}")

    def _update_detector_intrinsics_from_camera_info(self) -> None:
        """Update detector intrinsics from RGBD camera info when using ROS wrapper frames."""
        if self.detector is None or self.depth_camera_info is None:
            return

        k = getattr(self.depth_camera_info, "k", None)
        if k is None or len(k) < 6:
            return

        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        if fx > 0.0 and fy > 0.0:
            self.detector.fx = fx
            self.detector.fy = fy
            self.detector.cx = cx
            self.detector.cy = cy

    # Removed _schedule_next_detection to use periodic timer instead

    def _release_detector(self) -> None:
        """Cleanup detector and timer."""
        if self.loop_timer is not None:
            self.destroy_timer(self.loop_timer)
            self.loop_timer = None

        self.detection_period_sec = 0.0

        if self.detector is not None:
            try:
                self.detector.stop_camera()
            except Exception:
                pass
        self.detector = None
        self.detector_running = False

    def _get_available_model_paths(self) -> List[str]:
        """Return model files as relative paths matching the init request format."""
        module_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(module_root)
        model_dir = os.path.join(repo_root, "model")

        if not os.path.isdir(model_dir):
            return []

        supported_extensions = {".pt", ".pth", ".onnx", ".engine"}
        model_paths = []
        for filename in sorted(os.listdir(model_dir)):
            full_path = os.path.join(model_dir, filename)
            if not os.path.isfile(full_path):
                continue

            _, extension = os.path.splitext(filename)
            if extension.lower() not in supported_extensions:
                continue

            model_paths.append(os.path.join("model", filename))

        return model_paths

    def _publish_available_models(self) -> None:
        """Publish available model paths as a JSON array."""
        if not hasattr(self, "available_models_pub"):
            return

        payload = json.dumps(self._get_available_model_paths())
        self.available_models_pub.publish(String(data=payload))

    def _publish_status(self, level: str, message: str) -> None:
        """Publish status message."""
        payload = {"level": level, "message": message}
        self.status_pub.publish(String(data=json.dumps(payload)))

    def destroy_node(self) -> bool:
        """Cleanup on shutdown."""
        self._release_detector()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
