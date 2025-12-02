# train_yolo11_seg_advanced.py

from ultralytics import YOLO

import os
from pathlib import Path

# === 🔧 Pfad zu deinem exportierten Datensatz ===
base_dir = Path("imageset/Tennisball_seg_600.v2-tennisball-seg-dataset-607-v02.yolov11")

# === Unterordner, die geprüft werden sollen ===
subdirs = ["train/labels", "valid/labels", "test/labels"]  # ggf. anpassen

def has_only_boxes(label_path):
    """True, wenn Datei nur Box-Labels enthält (keine Segmente)."""
    try:
        with open(label_path, "r") as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]
        # Datei gilt als fehlerhaft, wenn eine Zeile <= 5 Werte hat
        return any(len(parts) <= 5 for parts in lines)
    except Exception as e:
        print(f"Fehler beim Lesen von {label_path}: {e}")
        return False





def main():
    # 1) Vortrainiertes YOLO11-Seg-Modell laden
    model = YOLO("yolo11m-seg.pt")  # „m“ (medium) liefert meist bessere Ergebnisse als „n“ (nano)
    # 2) Trainingsparameter
    data_yaml = "imageset/Tennisball_seg_600.v2-tennisball-seg-dataset-607-v02.yolov11/data.yaml"

    results = model.train(
        data=data_yaml,
        epochs=250,       
        imgsz=640,
        batch=8,
        device="0",
        name="tennisball_600_seg_yolo11_v02",
        exist_ok=True,

        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=15.0, translate=0.1, scale=0.25, shear=2.0,
        flipud=0.0, fliplr=0.5,
        mosaic=0.5, mixup=0.1,  

        patience=30,
        optimizer='AdamW',
        lr0=0.001, lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=15,
        
    )


    # 3) Validierung
    metrics = model.val()
    print(metrics)
    #model.val(data="test_data.yaml")

    # 4) Bestes Modell speichern
    model.save("tennisball_600_seg_yolo11_v02.pt")

if __name__ == "__main__":
    
    total_deleted = 0
    for sub in subdirs:
        dir_path = base_dir / sub
        if not dir_path.exists():
            continue

        for file in dir_path.glob("*.txt"):
            if has_only_boxes(file):
                print(f"❌ Entferne fehlerhafte Label-Datei: {file}")
                os.remove(file)
                total_deleted += 1

    print(f"\n✅ Fertig! {total_deleted} fehlerhafte Label-Dateien gelöscht.")

    main()
    
