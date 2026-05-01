import argparse
import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ros2_detection_interfaces.msg import DetectionArray

from ros2_detection_interfaces.srv import Init, Release, Start, Stop


class DetectorClient(Node):
    def __init__(self) -> None:
        super().__init__("detector_client")

        self.init_client = self.create_client(Init, "/detector/init")
        self.start_client = self.create_client(Start, "/detector/start")
        self.stop_client = self.create_client(Stop, "/detector/stop")
        self.release_client = self.create_client(Release, "/detector/release")

        self.status_sub = self.create_subscription(String, "/detector/status", self._on_status, 10)
        self.model_info_sub = self.create_subscription(String, "/detector/model_info", self._on_model_info, 10)
        self.results_sub = self.create_subscription(DetectionArray, "/detector/results", self._on_results, 2)
    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        start = time.time()
        clients = [self.init_client, self.start_client, self.stop_client, self.release_client]

        while time.time() - start < timeout_sec:
            if all(client.service_is_ready() for client in clients):
                return True
            for client in clients:
                client.wait_for_service(timeout_sec=0.2)
            rclpy.spin_once(self, timeout_sec=0.0)

        return False

    def call_init(self, model_path: str, classes: list, use_realsense_ros_wrapper: bool, confidence: float) -> bool:
        req = Init.Request()
        req.model_type = "yolo"
        req.model_path = model_path
        req.classes = classes
        req.use_realsense_ros_wrapper = use_realsense_ros_wrapper
        req.confidence = confidence
        req.use_decimation = False
        req.use_spatial = False
        req.use_temporal = True
        req.use_hole_filling = True
        req.use_mask_filter = True
        req.color_resolution_width = 640
        req.color_resolution_height = 480
        req.fps = 30
        req.rcnn_class_names = []

        future = self.init_client.call_async(req)
        return self._wait_future(future, "init")

    def call_start(self) -> bool:
        req = Start.Request()
        future = self.start_client.call_async(req)
        return self._wait_future(future, "start")

    def call_stop(self) -> bool:
        req = Stop.Request()
        future = self.stop_client.call_async(req)
        return self._wait_future(future, "stop")

    def call_release(self) -> bool:
        req = Release.Request()
        future = self.release_client.call_async(req)
        return self._wait_future(future, "release")

    def _wait_future(self, future, name: str, timeout_sec: float = 20.0) -> bool:
        start = time.time()
        while not future.done() and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            self.get_logger().error(f"Service '{name}' timed out")
            return False

        response = future.result()
        if response is None:
            self.get_logger().error(f"Service '{name}' returned no response")
            return False

        ok = bool(getattr(response, "success", False))
        msg = getattr(response, "message", "")
        if ok:
            self.get_logger().info(f"Service '{name}' OK: {msg}")
        else:
            self.get_logger().error(f"Service '{name}' FAILED: {msg}")

        if hasattr(response, "model_info") and response.model_info:
            self.get_logger().info(f"Model info: {response.model_info}")

        return ok

    def _on_status(self, msg: String) -> None:
        self.get_logger().info(f"STATUS: {msg.data}")

    def _on_model_info(self, msg: String) -> None:
        self.get_logger().info(f"MODEL_INFO: {msg.data}")

    def _on_results(self, msg: DetectionArray) -> None:

        # Header aus ROS Message
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id

        num = len(msg.detections)
        print("#" * 40)
        print(f"RESULTS: num_detections={num}")


        for idx, det in enumerate(msg.detections, start=1):

            label = det.label
            conf = det.confidence

            median_xyz = det.median_xyz
            object_center = det.object_center

            conf_str = f"{float(conf):.3f}" if conf is not None else "n/a"

            if median_xyz is not None and object_center is not None:

                print(
                    f"  - #{idx} class={label} conf={conf_str} "
                    f"m_x={median_xyz.x:.3f} m_y={median_xyz.y:.3f} m_z={median_xyz.z:.3f} "
                    f"c_x={object_center.x:.3f} c_y={object_center.y:.3f} c_z={object_center.z:.3f}"
                )

            else:
                print(
                    f"  - #{idx} class={label} conf={conf_str} x=n/a y=n/a z=n/a"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Service-based client for ros2_detection")
    parser.add_argument("action", choices=["run", "init", "start", "stop", "release"], help="Action to perform")
    parser.add_argument("--model-path", default="model/tennisball_600_seg_yolo11_v02.pt", help="YOLO model path")
    parser.add_argument("--classes", nargs="+", default=[], help="Classes to detect")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--duration", type=float, default=10.0, help="Listen duration in seconds for action=run")
    parser.add_argument("--keep-running", action="store_true", help="Do not auto stop/release after run duration")
    parser.add_argument("--use-realsense-ros-wrapper", action="store_true", default=False, help="Use RealSense ROS wrapper")
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = DetectorClient()

    try:
        if not node.wait_for_services(timeout_sec=15.0):
            node.get_logger().error("Detector services are not available. Is detector_node running?")
            return

        if args.action == "init":
            node.call_init(args.model_path, args.classes, args.use_realsense_ros_wrapper, args.confidence)
            return

        if args.action == "start":
            node.call_start()
            return

        if args.action == "stop":
            node.call_stop()
            return

        if args.action == "release":
            node.call_release()
            return

        if not node.call_init(args.model_path, args.classes, args.use_realsense_ros_wrapper, args.confidence):
            return

        if not node.call_start():
            return

        end_time: Optional[float] = None if args.duration <= 0 else (time.time() + args.duration)
        while rclpy.ok() and (end_time is None or time.time() < end_time):
            rclpy.spin_once(node, timeout_sec=0.2)

        if not args.keep_running:
            node.call_stop()
            node.call_release()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
