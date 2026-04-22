# Docker Deployment Guide

## Prerequisites

- **Docker Desktop** (Windows: https://www.docker.com/products/docker-desktop)
- **WSL 2** (Windows Subsystem for Linux 2, usually comes with Docker Desktop on Windows)
- **Git** (optional, for cloning)

## Building the Docker Image

### Option 1: Using Docker Compose (Recommended)

```bash
# In project root directory
docker-compose build
```

### Option 2: Using Docker Directly

```bash
docker build -t ros2_detector:latest .
```

## Running the Container

### Option 1: Using Docker Compose

```bash
# Start the container
docker-compose up -d

# View logs
docker-compose logs -f detector

# Stop the container
docker-compose down
```

### Option 2: Using Docker Directly

```bash
docker run -it \
  --name ros2_detector \
  -e ROS_DOMAIN_ID=0 \
  -v $(pwd)/tennisball_600_seg_yolo11_v02.pt:/root/project/tennisball_600_seg_yolo11_v02.pt:ro \
  -v $(pwd)/imageset:/root/project/imageset:ro \
  ros2_detector:latest
```

## Communicating with the Node

### From Host Machine

If you need to communicate with ROS2 topics from your host machine:

1. **Windows (Native Docker Desktop):**
   ```powershell
   # Install ROS2 on Windows or use Windows Python with rclpy
   # Then publish/subscribe to the same ROS_DOMAIN_ID (0)
   ```

2. **WSL 2 / Linux:**
   ```bash
   # Install ROS2 in WSL2
   sudo apt-get install ros-humble-desktop
   
   # In WSL2 terminal, subscribe to detector output:
   source /opt/ros/humble/setup.bash
   ros2 topic echo /detector/results
   
   # Publish detection request:
   ros2 topic pub /detector/request std_msgs/String "{data: '{\"classes\":[\"Tennisball\"],\"confidence\":0.45}'}"
   ```

### From Another Docker Container

```bash
# Run a second container on the same network
docker run --rm -it \
  --network ros2_network \
  -e ROS_DOMAIN_ID=0 \
  ros:humble bash

# Inside container:
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic echo /detector/status
```

## RealSense Camera Access

### Windows/Docker Desktop

RealSense access in Docker on Windows is limited. Recommended approaches:

**Option A: Run detector outside Docker, access via ROS2 over network**
- Host runs detector natively (no Docker)
- Docker container connects to host ROS2 network

**Option B: Use WSL 2 with Linux device passthrough**
```bash
# In docker-compose.yml, uncomment:
# devices:
#   - /dev/bus/usb

# Make sure docker has permission in WSL2:
# docker run --privileged ...
```

**Option C: Use IP-based communication**
- Publish/subscribe via network interface instead of IPC

## GPU Support (NVIDIA CUDA)

If you have an NVIDIA GPU:

1. Install **NVIDIA Container Runtime** on your host

2. Uncomment in `docker-compose.yml`:
   ```yaml
   runtime: nvidia
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
   ```

3. Or with docker run:
   ```bash
   docker run --gpus all ros2_detector:latest
   ```

## Volumes & Model Files

Models are mounted as read-only in the container:

```yaml
volumes:
  - ./model:/root/project/model:ro
```

Make sure these files exist in your project root before starting the container.

## Environmental Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROS_DOMAIN_ID` | `0` | ROS2 domain for multi-network isolation |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | Middleware implementation |

## Troubleshooting

### Container won't start on Windows

```bash
# Check docker logs
docker logs ros2_detector

# Rebuild with verbose output
docker build --no-cache -t ros2_detector:latest .
```

### Can't access topics from host

1. Check both host and container have same `ROS_DOMAIN_ID`
2. Verify network connectivity: `docker network inspect ros2_network`
3. Try publishing from container to verify internal communication works

### Model files not found

- Ensure `.pt` files exist in project root
- Check volume mounts: `docker inspect ros2_detector | grep -A 20 Mounts`

### RealSense camera not accessible

See "RealSense Camera Access" section above. For development on Windows, recommend:
1. Run detector natively (Python directly or WSL2)
2. Use Docker only for CI/CD or ROS2 middleware testing

## Example Workflow

### Terminal 1: Start container
```bash
docker-compose up
```

### Terminal 2 (WSL2): Monitor detector status
```bash
wsl
source /opt/ros/humble/setup.bash
ros2 topic echo /detector/status
```

### Terminal 3 (WSL2): Send configuration + request
```bash
wsl
source /opt/ros/humble/setup.bash

# Initialize
ros2 topic pub /detector/config std_msgs/String \
  "{data: '{\"action\":\"init\",\"model_type\":\"yolo\",\"model_path\":\"/root/project/tennisball_600_seg_yolo11_v02.pt\",\"classes\":[\"Tennisball\"],\"confidence\":0.5,\"camera\":{\"color_resolution\":[640,480],\"fps\":30}}'}"

# Request detection
ros2 topic pub /detector/request std_msgs/String \
  "{data: '{\"classes\":[\"Tennisball\"],\"confidence\":0.45,\"max_results\":5}'}"

# View results
ros2 topic echo /detector/results
```

### Cleanup
```bash
docker-compose down
```

## Performance Considerations

- **CPU-only**: Expect ~2-5 FPS depending on CPU
- **GPU (CUDA)**: First run will download cuDNN (~2GB), plan accordingly
- **Memory**: Ensure Docker Desktop has ≥8GB RAM allocated
- **Storage**: Image size ~2-3GB depending on base OS and dependencies

## Building for Production

For a smaller, optimized image:

```dockerfile
# Multi-stage build (add to Dockerfile)
FROM ros:humble as builder
# ... build steps ...

FROM ros:humble-slim
# Copy only necessary files
COPY --from=builder /root/ros2_ws/install /root/ros2_ws/install
```
