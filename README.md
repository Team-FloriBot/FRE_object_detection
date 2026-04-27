# FRE Object Detection

This project provides a pipeline for synthetic data generation, YOLO model training, and object detection with RealSense cameras, optimized for both PC and NVIDIA Jetson.

## Workflows

### PC (Laptop with NVIDIA GPU)

1. **Generate Synthetic Data:**
   ```bash
   docker compose --profile gen-pc up --build
   ```

2. **Train Model:**
   ```bash
   docker compose --profile train-pc up --build
   ```

3. **Run Detection:**
   ```bash
   docker compose --profile detect-pc up --build
   ```

### Jetson (NVIDIA JetPack)

1. **Generate Synthetic Data:**
   ```bash
   docker compose --profile gen-jetson up --build
   ```

2. **Train Model:**
   ```bash
   docker compose --profile train-jetson up --build
   ```

3. **Export TensorRT (Highly recommended):**
   ```bash
   docker compose --profile export-jetson up --build
   ```
   *Note: Exporting must be done on the Jetson itself.*

4. **Run Detection:**
   ```bash
   docker compose --profile detect-jetson up --build
   ```

## Model Information
The detection script (`detection/detection.py`) automatically looks for a `.engine` (TensorRT) file in the `model/` directory. If not found, it falls back to the `.pt` (PyTorch) model.
