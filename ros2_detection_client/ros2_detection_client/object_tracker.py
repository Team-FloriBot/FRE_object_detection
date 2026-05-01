import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, Point
from rclpy.time import Time
from std_msgs.msg import Bool

from ros2_detection_interfaces.msg import DetectionArray, TrackedObject, TrackedObjectArray
import math

def distance(self, p1, p2):
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2
    )

THRESHOLD = 0.3

class ObjectTracker(Node):
    def __init__(self) -> None:
        super().__init__("object_tracker")

        self.to_frame_rel = "map"


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.results_sub = self.create_subscription(DetectionArray, "/detector/results", self._on_detection_results, 2)
        self.tracked_objects_publisher = self.create_publisher(TrackedObjectArray, "/tracker/tracked_objects", 10)
        self.set_active = self.create_subscription(Bool, "/tracker/active", self._on_active, 10)
        self.active = False

        self.object_map = {}  # Map: object_id -> (label, last_seen_time, last_position)

        self.next_id = 0

        self.labels = []

    def _on_active(self, msg: Bool) -> None:
        
        self.active = msg.data

    def _on_detection_results(self, msg: DetectionArray) -> None:
        if not self.active:
            return
        
        # Header aus ROS Message
        target_time = msg.header.stamp
        target_frame = msg.header.frame_id

        # TF lookup map <- camera(frame aus message)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.to_frame_rel,
                target_frame,
                target_time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        num = len(msg.detections)

        labels = msg.searched_classes

        for label in labels:
            if label not in self.labels:
                self.labels.append(label)

        for idx, det in enumerate(msg.detections, start=1):

            conf = det.confidence
            det_id = det.id

            object_center = det.object_center

            # Punkt im Kamera-KS
            point_cam = PointStamped()
            point_cam.header.frame_id = target_frame
            point_cam.header.stamp = target_time
            point_cam.point = object_center

            # Transform nach Map
            point_map = do_transform_point(point_cam, transform)

            best_id = None
            best_dist = float("inf")


            for obj_id, obj in self.object_map.items():
                last_pos = obj["pos"]
                if last_pos is None:
                    continue

                dist = distance(last_pos, point_map.point)

                if dist < THRESHOLD and dist < best_dist:
                    best_dist = dist
                    best_id = obj_id

            matched_id = best_id
  
            alpha = 0.3

            if matched_id is not None:
                obj = self.object_map[matched_id]
                old = obj["pos"]

                smoothed = Point()
                smoothed.x = (1 - alpha) * old.x + alpha * point_map.point.x
                smoothed.y = (1 - alpha) * old.y + alpha * point_map.point.y
                smoothed.z = (1 - alpha) * old.z + alpha * point_map.point.z

                obj["label_counts"][det.label] = obj["label_counts"].get(det.label, 0) + conf
                if obj["counter"] >= 8:
                    obj["label"] = max(obj["label_counts"], key=obj["label_counts"].get)
                obj["pos"] = smoothed
                obj["last_seen"] = target_time
                obj["counter"] += 1
                obj["last_det_id"] = det_id
          
            else:
                new_id = self.next_id
                self.next_id += 1

                self.object_map[new_id] = {
                    "id": new_id,
                    "label": None,
                    "label_counts": {det.label: conf},
                    "last_seen": target_time,
                    "pos": point_map.point,
                    "counter": 1,
                    "last_det_id": det_id
                }
   
        msg_out = TrackedObjectArray()
        msg_out.header.stamp = target_time
        msg_out.header.frame_id = self.to_frame_rel

        current_t = Time.from_msg(msg.header.stamp)

        to_delete = []
        for obj_id, obj in self.object_map.items():

            if obj["label"] is None:
                previous_t = Time.from_msg(obj["last_seen"])
                duration = current_t - previous_t
                if duration.nanoseconds > 10e9:  # 5e9 Nanosekunden = 5 Sekunden
                    to_delete.append(obj_id)
                continue
  
            tracked = TrackedObject()
            tracked.id = obj_id
            tracked.label = obj["label"]

            tracked.position.x = obj["pos"].x
            tracked.position.y = obj["pos"].y
            tracked.position.z = obj["pos"].z

            msg_out.objects.append(tracked)

        
        for obj_id in to_delete:
            del self.object_map[obj_id]

        self.tracked_objects_publisher.publish(msg_out)



def main(args=None) -> None:
    rclpy.init(args=args)

    node = ObjectTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()