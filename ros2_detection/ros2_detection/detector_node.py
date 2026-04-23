import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from ros2_detection_interfaces.srv import Init, Release, Start, Stop
from std_msgs.msg import String
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD
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

        self.detections_counter = 0
        self.debug_image_dir = os.environ.get("DETECTION_DEBUG_DIR", "/root/ros2_ws/debug_images")
        os.makedirs(self.debug_image_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.color_img = None
        self.depth_img = None
        self.depth_camera_info = None
        self._last_waiting_frame_status_sec = 0.0

        self.sub = self.create_subscription(
            RGBD,
            "/camera/rgbd/image",
            self.realsense_callback,
            10
        )

        self._publish_status("ready", "Detector node started. Waiting for /detector/init.")

    def realsense_callback(self, msg):
        self.color_img = self.bridge.imgmsg_to_cv2(msg.color, "bgr8")
        self.depth_img = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        self.depth_camera_info = getattr(msg, "depth_camera_info", None)

        print("RGBD:", self.color_img.shape, self.depth_img.shape)

    def get_realsense_images(self):
        return self.color_img, self.depth_img

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
                    "use_realsense_ros_wrapper": bool(request.use_realsense_ros_wrapper),
                    "color_resolution": [request.color_resolution_width, request.color_resolution_height],
                    "fps": int(request.fps),
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

            frames = self.detector.get_frame()
            if self.detector.get_realsense_images is not None:
                color_image, depth_image = frames

                # ROS wrapper can briefly return no images at startup.
                if color_image is None or depth_image is None:
                    now = time.monotonic()
                    if now - self._last_waiting_frame_status_sec > 2.0:
                        self._publish_status("ready", "Waiting for RGBD frames on /camera/rgbd/image.")
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
            for obj_id, data in fused_objs_data.items():
                label = data.get("class")
                confidence = data.get("confidence", None)
                median_xyz = data.get("median_xyz", None)
                object_center = data.get("object_center", None)

                entry = {
                    "id": obj_id,
                    "class": label,
                    "confidence": confidence,
                    "median_xyz": median_xyz,
                    "object_center": object_center,
                }

                detections.append(entry)

            if detections:
                self.detections_counter += 1
            if self.detections_counter <= 5:
                cv2.imwrite(os.path.join(self.debug_image_dir, f"detection_before_debug{self.detections_counter}.jpg"), color_image)
                cv2.imwrite(os.path.join(self.debug_image_dir, f"detection_debug{self.detections_counter}.jpg"), annotated_image)

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

    def _update_detector_intrinsics_from_camera_info(self) -> None:
        """Update detector intrinsics from RGBD camera info when using ROS wrapper frames."""
        if self.detector is None or self.depth_camera_info is None:
            return

        k = getattr(self.depth_camera_info, "k", None)
        if not k or len(k) < 6:
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
