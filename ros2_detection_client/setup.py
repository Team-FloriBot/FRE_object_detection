from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_detection_client'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aaron Stark',
    maintainer_email='aaron.stark@example.com',
    description='ROS2 detector client node for YOLO object detection with RealSense depth fusion',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_client = ros2_detection_client.detector_client:main',
            'object_tracker = ros2_detection_client.object_tracker:main',
        ],
    },
)
