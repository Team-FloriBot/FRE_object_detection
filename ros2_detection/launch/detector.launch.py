from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    detector_node = Node(
        package='ros2_detection',
        executable='detector_node',
        name='detector_node',
        output='screen',
    )
    
    return LaunchDescription([
        detector_node,
    ])
