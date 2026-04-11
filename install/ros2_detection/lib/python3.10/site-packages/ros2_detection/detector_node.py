import json
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .detection_model_selection import ObjDetection


class DetectionNode(Node):
    """
    ROS2 node that wraps ObjDetection and exposes a topic-based interface.

    Subscriptions:
      - /detector/config (std_msgs/String with JSON)
      - /detector/request (std_msgs/String with JSON)

    Publications:
      - /detector/model_info (std_msgs/String with JSON)
      - /detector/results (std_msgs/String with JSON)
      - /detector/status (std_msgs/String with JSON)
    """

    def __init__(self) -> None:
        super().__init__("detector_node")

        self.detector: Optional[ObjDetection] = None
        self.detector_running = False

        self.config_sub = self.create_subscription(
            String, "/detector/config", self._on_config, 10
        )
        self.request_sub = self.create_subscription(
            String, "/detector/request", self._on_request, 10
        )

        self.model_info_pub = self.create_publisher(String, "/detector/model_info", 10)
        self.results_pub = self.create_publisher(String, "/detector/results", 10)
        self.status_pub = self.create_publisher(String, "/detector/status", 10)

        self._publish_status("ready", "Detector node started. Waiting for /detector/config.")

    def _on_config(self, msg: String) -> None:
        payload = self._parse_json(msg.data, "/detector/config")
        if payload is None:
            return

        action = str(payload.get("action", "init")).lower()

        if action == "init":
            self._init_detector(payload)
            return

        if action == "start":
            if self.detector is None:
                self._publish_status("error", "Cannot start detector: not initialized.")
                return
            self.detector_running = True
            self._publish_status("ok", "Detector started.")
            return

        if action == "stop":
            self.detector_running = False
            self._publish_status("ok", "Detector stopped.")
            return

        if action == "release":
            self._release_detector()
            self._publish_status("ok", "Detector released.")
            return

        self._publish_status("error", f"Unknown config action: {action}")

    def _on_request(self, msg: String) -> None:
        payload = self._parse_json(msg.data, "/detector/request")
        if payload is None:
            return

        if self.detector is None:
            self._publish_status("error", "Detector not initialized. Send /detector/config first.")
            return

        if not self.detector_running:
            self._publish_status("error", "Detector is stopped. Send action=start on /detector/config.")
            return

        requested_classes = payload.get("classes")
        requested_conf = payload.get("confidence")
        max_results = int(payload.get("max_results", 0))

        if isinstance(requested_conf, (int, float)):
            self.detector.conf = float(requested_conf)

        searched_classes = self._apply_requested_classes(requested_classes)

        try:
            frames = self.detector.get_frame()
            aligned_frames = self.detector.align_frames(frames)
            color_image, depth_image = self.detector.depth_filter(aligned_frames)

            if color_image is None or depth_image is None:
                self._publish_status("error", "No valid color/depth frame available.")
                return

            obj_masks, _ = self.detector.detect_obj(color_image)
            point_clouds, _ = self.detector.fuse(color_image, depth_image, obj_masks)

            detections = []
            for obj in obj_masks:
                label = obj.get("class")
                conf = obj.get("confidence", None)
                center_xy = obj.get("center", None)
                entry = {
                    "class": label,
                    "confidence": conf,
                    "center_xy": [int(center_xy[0]), int(center_xy[1])] if center_xy else None,
                    "median_xyz": None,
                    "balliness": None,
                    "radius": None,
                }

                for pc_obj in point_clouds.values():
                    if pc_obj.get("label") == label:
                        median = pc_obj.get("median", None)
                        if median is not None:
                            entry["median_xyz"] = [
                                float(median[0]),
                                float(median[1]),
                                float(median[2]),
                            ]
                        if "balliness" in pc_obj:
                            entry["balliness"] = float(pc_obj["balliness"])
                        if "radius" in pc_obj:
                            entry["radius"] = float(pc_obj["radius"])
                        break

                detections.append(entry)

            if max_results > 0:
                detections = detections[:max_results]

            result_payload = {
                "ok": True,
                "searched_classes": searched_classes,
                "model": self.detector.get_model_info(),
                "num_detections": len(detections),
                "detections": detections,
            }
            self.results_pub.publish(String(data=json.dumps(result_payload)))
            self._publish_status("ok", f"Published {len(detections)} detections.")

        except Exception as exc:
            self._publish_status("error", f"Detection request failed: {exc}")

    def _init_detector(self, payload: Dict[str, Any]) -> None:
        model_type = str(payload.get("model_type", "yolo")).lower()
        model_path = payload.get("model_path")
        classes = payload.get("classes", ["Tennisball"])
        confidence = float(payload.get("confidence", 0.5))
        rcnn_class_names = payload.get("rcnn_class_names")

        filters = payload.get("filters", {})
        use_decimation = bool(filters.get("use_decimation", False))
        use_spatial = bool(filters.get("use_spatial", False))
        use_temporal = bool(filters.get("use_temporal", True))
        use_hole_filling = bool(filters.get("use_hole_filling", True))
        use_mask_filter = bool(filters.get("use_mask_filter", True))

        camera = payload.get("camera", {})
        color_resolution = tuple(camera.get("color_resolution", [640, 480]))
        fps = int(camera.get("fps", 30))

        if not isinstance(classes, list) or len(classes) == 0:
            self._publish_status("error", "Config field 'classes' must be a non-empty list.")
            return

        self._release_detector()

        try:
            self.detector = ObjDetection(
                classes=classes,
                model_type=model_type,
                model_path=model_path,
                rcnn_class_names=rcnn_class_names,
                use_decimation=use_decimation,
                use_spatial=use_spatial,
                use_temporal=use_temporal,
                use_hole_filling=use_hole_filling,
                use_mask_filter=use_mask_filter,
                conf=confidence,
            )
            self.detector.initialize_realsense(color_resolution=color_resolution, fps=fps)
            self.detector_running = True

            model_info = self.detector.get_model_info()
            self.model_info_pub.publish(String(data=json.dumps(model_info)))
            self._publish_status("ok", "Detector initialized and started.")

        except Exception as exc:
            self.detector = None
            self.detector_running = False
            self._publish_status("error", f"Failed to initialize detector: {exc}")

    def _release_detector(self) -> None:
        if self.detector is not None:
            try:
                self.detector.stop_camera()
            except Exception:
                pass
        self.detector = None
        self.detector_running = False

    def _apply_requested_classes(self, classes: Any) -> List[str]:
        if self.detector is None:
            return []

        if not isinstance(classes, list) or len(classes) == 0:
            return list(self.detector.classes)

        if self.detector.model_type == "yolo":
            model_names = self.detector.model.names
            class_ids = [idx for idx, name in model_names.items() if name in classes]
            self.detector.class_ids = class_ids
            self.detector.classes = [model_names[idx] for idx in class_ids]
            return list(self.detector.classes)

        id_to_name = self.detector.id_to_name
        class_ids = [idx for idx, name in id_to_name.items() if name in classes]
        self.detector.class_ids = class_ids
        self.detector.classes = [id_to_name[idx] for idx in class_ids]
        return list(self.detector.classes)

    def _publish_status(self, level: str, message: str) -> None:
        payload = {"level": level, "message": message}
        self.status_pub.publish(String(data=json.dumps(payload)))

    def _parse_json(self, raw: str, source_topic: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._publish_status("error", f"Invalid JSON on {source_topic}: {exc}")
            return None

        if not isinstance(payload, dict):
            self._publish_status("error", f"JSON on {source_topic} must be an object.")
            return None

        return payload

    def destroy_node(self) -> bool:
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
