#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "[1/4] Listing detector topics"
ros2 topic list | grep -E '^/detector/(config|request|model_info|results|status)$' || true

echo "[2/4] Publishing one-shot init config"
ros2 topic pub -1 /detector/config std_msgs/msg/String "{data: '$(tr -d '\n' < ros2_detection/example_config.json)'}"

echo "[3/4] Waiting for one status message"
ros2 topic echo --once /detector/status

echo "[4/4] Waiting for model info (if init succeeded)"
ros2 topic echo --once /detector/model_info || true

echo "Smoke test done."
