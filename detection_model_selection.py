"""Backward-compatible import path for ObjDetection.

Keep this thin wrapper so pre-ROS scripts can continue to use:
    from detection_model_selection import ObjDetection
"""

try:
    # Installed package layout.
    from ros2_detection.detection_model_selection import ObjDetection
except ImportError:
    # Source tree layout (workspace root).
    from ros2_detection.ros2_detection.detection_model_selection import ObjDetection

__all__ = ["ObjDetection"]
