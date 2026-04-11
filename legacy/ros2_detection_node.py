"""Legacy ROS2 node entrypoint kept for compatibility.

Use the package entrypoint:
    ros2 run ros2_detection detector_node
"""

from ros2_detection.ros2_detection.detector_node import main


if __name__ == "__main__":
    main()
