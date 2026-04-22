import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD

#ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true 

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__("realsense_sub")
        # sensor_msgs/Image -> OpenCV Bilder konvertieren
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            RGBD,
            "/camera/rgbd/image",
            self.callback,
            10
        )

    def callback(self, msg):
        color = self.bridge.imgmsg_to_cv2(msg.color, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")

        print("RGBD:", color.shape, depth.shape)


def main():
    rclpy.init()
    node = RealSenseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
