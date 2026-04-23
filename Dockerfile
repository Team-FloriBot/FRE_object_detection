# Use ROS2 Humble as base image
FROM ros:humble

# Set working directory
WORKDIR /root/ros2_ws

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    libgl1 \
    libusb-1.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch

RUN pip install --no-cache-dir \
    opencv-python \
    scikit-learn \
    pyrealsense2 \
    pyyaml \
    ultralytics

RUN pip install --no-cache-dir "numpy<2.0"

RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-realsense2-camera-msgs


# Create directory structure
RUN mkdir -p /root/ros2_ws/src

# Copy the project files
COPY ros2_detection /root/ros2_ws/src/ros2_detection
COPY ros2_detection_interfaces /root/ros2_ws/src/ros2_detection_interfaces
COPY ros2_detection_client /root/ros2_ws/src/ros2_detection_client
COPY model /root/ros2_ws/model
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

ENV ULTRALYTICS_CONFIG_DIR=/tmp

# Source setup in bashrc for convenience
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Set environment variables
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

