import json
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from ros2_detection_interfaces.srv import Init, Release, Start, Stop
from std_msgs.msg import String

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
      - /detector/model_info (std_msgs/String with JSON)
      - /detector/results (std_msgs/String with JSON)
      - /detector/status (std_msgs/String with JSON)
    """

    def __init__(self) -> None:
        super().__init__("detector_node")

        self.detector: Optional[ObjDetection] = None
        self.detector_running = False
        self.loop_timer = None
        self.detection_period_sec = 0.0
        self.current_config: Dict[str, Any] = {}

        # Publishers for monitoring
        self.model_info_pub = self.create_publisher(String, "/detector/model_info", 10)
        self.results_pub = self.create_publisher(String, "/detector/results", 10)
        self.status_pub = self.create_publisher(String, "/detector/status", 10)

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

        self._publish_status("ready", "Detector node started. Waiting for /detector/init.")

    def _handle_init(self, request, response) -> None:
        """Initialize detector with given configuration."""
        try:
            # Build config dict from service request
            config = {
                "model_type": request.model_type,
                "model_path": request.model_path,
                "classes": list(request.classes),
                "confidence": float(request.confidence),
                "rcnn_class_names": list(request.rcnn_class_names) if request.rcnn_class_names else None,
                "filters": {
                    "use_decimation": bool(request.use_decimation),
                    "use_spatial": bool(request.use_spatial),
                    "use_temporal": bool(request.use_temporal),
                    "use_hole_filling": bool(request.use_hole_filling),
                    "use_mask_filter": bool(request.use_mask_filter),
                },
                "camera": {
                    "color_resolution": [request.color_resolution_width, request.color_resolution_height],
                    "fps": int(request.fps),
                },
            }

            if not isinstance(config["classes"], list) or len(config["classes"]) == 0:
                response.success = False
                response.message = "Config field 'classes' must be a non-empty list."
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
                conf=config["confidence"],
            )

            # Initialize camera
            self.detector.initialize_realsense(
                color_resolution=tuple(config["camera"]["color_resolution"]),
                fps=config["camera"]["fps"],
            )

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
        """Start adaptive detection loop."""
        if self.detector is None:
            response.success = False
            response.message = "Detector not initialized. Call /detector/init first."
            return response

        if self.detector_running:
            response.success = False
            response.message = "Detector already running."
            return response

        self.detector_running = True

        # Use adaptive scheduling: each cycle waits only for the remaining
        # time after the last inference step finished.
        fps = int(self.current_config.get("camera", {}).get("fps", 30))
        if fps <= 0:
            fps = 30
        self.detection_period_sec = 1.0 / fps  # seconds
        self._schedule_next_detection(0.0)

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
        """Run one detection cycle and schedule the next one adaptively."""
        if self.detector is None or not self.detector_running:
            return

        cycle_start = time.monotonic()
        try:
            current_timer = self.loop_timer
            self.loop_timer = None
            if current_timer is not None:
                current_timer.cancel()
                self.destroy_timer(current_timer)

            detect_start = time.monotonic()
            frames = self.detector.get_frame()
            aligned_frames = self.detector.align_frames(frames)
            color_image, depth_image = self.detector.depth_filter(aligned_frames)

            if color_image is None or depth_image is None:
                self._publish_status("error", "No valid color/depth frame available.")
                return
            
            objs_data, _ = self.detector.detect_obj(color_image)

            fused_objs_data, _ = self.detector.fuse(color_image, depth_image, objs_data)

            detections = []
            for obj_id, data in fused_objs_data.items():
                label = data.get("class")
                confidence = data.get("confidence", None)
                median_xyz = data.get("median_xyz", None)

                entry = {
                    "id": obj_id,
                    "class": label,
                    "confidence": confidence,
                    "median_xyz": median_xyz,
                }

                detections.append(entry)

            result_payload = {
                "ok": True,
                "searched_classes": self.detector.classes,
                "model": self.detector.get_model_info(),
                "num_detections": len(detections),
                "detections": detections,
            }
            self.results_pub.publish(String(data=json.dumps(result_payload)))

        except Exception as exc:
            self._publish_status("error", f"Detection loop error: {exc}")
        finally:
            if self.detector_running and self.detector is not None:
                elapsed_sec = time.monotonic() - cycle_start
                remaining_sec = max(0.0, self.detection_period_sec - elapsed_sec)
                self._schedule_next_detection(remaining_sec)

    def _schedule_next_detection(self, delay_sec: float) -> None:
        """Schedule the next detection cycle after the requested delay."""
        if self.detector is None or not self.detector_running:
            return

        if self.loop_timer is not None:
            self.destroy_timer(self.loop_timer)
            self.loop_timer = None

        self.loop_timer = self.create_timer(max(0.0, float(delay_sec)), self._detection_loop)

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
