# train_yolo11_seg_advanced.py

from ultralytics import YOLO

def main():
    # 1) Vortrainiertes YOLO11-Seg-Modell laden
    model = YOLO("yolo11m-seg.pt")  # „m“ (medium) liefert meist bessere Ergebnisse als „n“ (nano)
    #model = YOLO("yolo11_banane_apfel_orange-seg.pt")
    # 2) Trainingsparameter
    data_yaml = "dataset_banane_apfel_orange/data.yaml"

    results = model.train(
        data=data_yaml,
        epochs=300,       
        imgsz=640,
        batch=8,
        device="0",
        name="yolo11_banane_apfel_orange_seg2",
        exist_ok=True,

        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=15.0, translate=0.1, scale=0.5, shear=2.0,
        flipud=0.0, fliplr=0.5,
        mosaic=0.7, mixup=0.1,    # MixUp etwas erhöhen für kleine Klasse

        patience=30,
        optimizer='AdamW',
        lr0=0.001, lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=15,
        
    )


    # 3) Validierung
    metrics = model.val()


    # 4) Bestes Modell speichern
    model.save("yolo11_banane_apfel_orange2-seg.pt")

if __name__ == "__main__":
    main()
