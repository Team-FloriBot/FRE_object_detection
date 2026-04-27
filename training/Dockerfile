# Nutze das offizielle Ultralytics Image mit GPU-Support
FROM ultralytics/ultralytics:latest

# Arbeitsverzeichnis im Container festlegen
WORKDIR /usr/src/app

# System-Abhängigkeiten für OpenCV (noble-kompatibel)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Trainingsskript kopieren
COPY training/train.py ./training/train.py

# Installiere zusätzliche Python-Pakete falls nötig 
# (ultralytics ist bereits installiert)
RUN pip install --no-cache-dir numpy opencv-python pyyaml onnx onnxslim onnxruntime-gpu

# Befehl beim Start des Containers
CMD ["python3", "training/train.py"]