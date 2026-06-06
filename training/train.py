from pathlib import Path

import yaml
from ultralytics import YOLO


def _load_class_config(project_root: Path) -> tuple[int, list[str]]:
    candidate_files = [
        project_root / "dataset" / "data.yaml",
        project_root / "object_images" / "data.yaml",
    ]

    for source_yaml in candidate_files:
        print("looking for class config in:", source_yaml)
        if not source_yaml.exists():
            continue

        data = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names", [])
        print("loaded class config:", names)

        if isinstance(names, dict):
            names = [names[idx] for idx in sorted(names)]

        nc = int(data.get("nc", len(names)))
        print(nc)

        if names and nc == len(names):
            print(names)
            return nc, names

        raise ValueError(
            f"Ungueltige Klassenkonfiguration in {source_yaml}: nc={nc}, names={names}"
        )

    raise FileNotFoundError(
        "Keine gueltige class config gefunden (dataset/data.yaml oder object_images/data.yaml)."
    )


def _build_output_data_yaml(project_root: Path) -> Path:
    output_dir = project_root / "dataset"
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Erwarte dataset/images und dataset/labels fuer das Training.")

    nc, names = _load_class_config(project_root)
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }
    target_yaml = output_dir / "train_data.yaml"
    target_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print("using data yaml:", target_yaml)
    print("data yaml content:")
    print(target_yaml.read_text(encoding="utf-8"))
    return target_yaml


def main(name: str = "yolo26n_bee_beetle_butterfly") -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"

    model_path = project_root / "model" / "yolo26n-seg.pt"
    model = YOLO(str(model_path if model_path.exists() else "yolo26n-seg.pt"))

    data_yaml = _build_output_data_yaml(project_root)

    model.train(
        data=str(data_yaml),
        epochs=70, #100
        imgsz=640,
        batch=16,
        device="0",
        project=str(runs_dir),
        name=name,
        exist_ok=True,

        # --- REALISTISCHE FARBE ---
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.25,

        # --- REALISTISCHE GEOMETRIE ---
        degrees=5.0,
        translate=0.05,
        scale=0.35,
        shear=0.0,
        flipud=0.0,   # Top-Down Kamera
        fliplr=0.3,

        # --- MIX-STRATEGIEN ---
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        # --- TRAINING ---
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        close_mosaic=10,
    )


    model.val(
        data=str(data_yaml),
        project=str(runs_dir),
        name=f"{name}_val",
        exist_ok=True,
    )
    model.save(str(project_root / "model" / f"{name}-seg.pt"))


if __name__ == "__main__":
    main(name="yolo26n_soilspot")
