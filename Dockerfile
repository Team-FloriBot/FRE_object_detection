# Use ROS2 Humble as base image
FROM ros:humble

# Set working directory
WORKDIR /root/ros2_ws

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    libgl1 \
    libusb-1.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /root/ros2_ws/src

# Copy the project files
COPY ros2_detection /root/ros2_ws/src/ros2_detection

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchvision
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    scikit-learn \
    open3d \
    pyrealsense2 \
    matplotlib \
    pillow \
    pyyaml \
    requests \
    scipy \
    psutil \
    polars \
    ultralytics-thop
RUN pip install --no-cache-dir --no-deps ultralytics

# Build ROS2 workspace
WORKDIR /root/ros2_ws
RUN . /opt/ros/humble/setup.sh && colcon build --packages-select ros2_detection

# Source setup in bashrc for convenience
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Set environment variables
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Default command
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /root/ros2_ws/install/setup.bash && ros2 run ros2_detection detector_node"]
