FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Arbeitsverzeichnis
WORKDIR /app

# Requirements installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Script kopieren
COPY generate_synthetic/generate_synthetic.py ./generate_synthetic.py

# Standard-Ordner erzeugen
RUN mkdir -p object_images/train/images \
    && mkdir -p object_images/train/labels \
    && mkdir -p backgrounds \
    && mkdir -p dataset/images \
    && mkdir -p dataset/labels

# Startscript
COPY generate_synthetic/run.sh ./run.sh
RUN chmod +x run.sh

CMD ["./run.sh"]
