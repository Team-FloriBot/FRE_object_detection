# Use ROS2 Jazzy as base image for the client
FROM ros:jazzy

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

# Install ROS2 Jazzy specific dependencies
RUN apt-get update && apt-get install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /root/ros2_ws/src

# Copy the necessary packages
COPY ros2_detection_interfaces /root/ros2_ws/src/ros2_detection_interfaces
COPY ros2_detection_client /root/ros2_ws/src/ros2_detection_client

# Build the workspace
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

# Source setup in bashrc for convenience
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Set environment variables for compatibility with Humble
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Default command
CMD ["bash"]
