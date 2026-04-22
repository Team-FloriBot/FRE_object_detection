Entsprechendes Image für ROS

NVIDIA L4T ROS2 Humble Image
FROM nvcr.io/nvidia/ros:humble-ros-base-l4t-r35.2.1



Dockerfile ändern (nicht pip zum installieren --> jetson optimierte installation benutzen)

FROM nvcr.io/nvidia/ros:humble-ros-base-l4t-r35.2.1
WORKDIR /root/ros2_ws
# System deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    libgl1 \
    libusb-1.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
# Jetson-optimiertes PyTorch installieren
RUN pip install --no-cache-dir \
    torch==1.13.0 \
    torchvision==0.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116
# Python libs
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    scikit-learn \
    open3d \
    pyrealsense2 \
    ultralytics
COPY ros2_detection /root/ros2_ws/src/ros2_detection
COPY model /root/ros2_ws/model
RUN . /opt/ros/humble/setup.sh && colcon build --packages-select ros2_detection
CMD ["bash"]


docker-compose.yml ändern

deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker


TensorRT nutzen (optimiert)

Base‑Image = L4T (NVIDIA)
--gpus all gesetzt ist
NVIDIA Runtime aktiv ist
das Modell vorher exportieren:
yolo export model=yolov8n-seg.pt format=engine device=0 half=True
model = YOLO("yolov8n-seg.engine")


RealSense

devices:
  - /dev/video0:/dev/video0
  - /dev/bus/usb:/dev/bus/usb
