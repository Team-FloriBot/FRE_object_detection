# Nutze das offizielle Ultralytics Image mit GPU-Support
FROM ultralytics/ultralytics:latest

# Arbeitsverzeichnis im Container festlegen
WORKDIR /usr/src/app

# System-Abhängigkeiten für OpenCV und USB-Geräte (RealSense)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

    # Installiere zusätzliche Python-Pakete falls nötig 
# (ultralytics ist bereits installiert)
RUN pip install --no-cache-dir numpy opencv-python pyyaml

# Installiere Python-Pakete
RUN pip install --no-cache-dir pyrealsense2

# Trainingsskript kopieren
COPY detection/detection.py ./detection/detection.py

# Befehl beim Start des Containers
CMD ["python3", "detection/detection.py"]