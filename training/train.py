from pathlib import Path

import yaml
from ultralytics import YOLO


TRAINING_PRESETS = {
    "default": {
        "epochs": 70,
        "batch": 16,
        "patience": 20,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "hsv_h": 0.015,
        "hsv_s": 0.35,
        "hsv_v": 0.25,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.35,
        "shear": 0.0,
        "flipud": 0.0,
        "fliplr": 0.3,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    },
    # Fuer kleine echte Datensaetze: ca. 20-50 Bilder.
    "small_finetune": {
        "epochs": 140,
        "batch": 4,
        "patience": 35,
        "optimizer": "AdamW",
        "lr0": 0.00025,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.18,
        "degrees": 3.0,
        "translate": 0.03,
        "scale": 0.20,
        "shear": 0.0,
        "flipud": 0.0,
        "fliplr": 0.2,
        "mosaic": 0.15,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 20,
    },
}


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


def _resolve_model_path(project_root: Path, model_name: str) -> Path | str:
    if model_name == "yolo26n-seg.pt":
        local_model_path = project_root / "model" / model_name
        return local_model_path if local_model_path.exists() else model_name

    model_path = project_root / "model" / model_name
    if model_path.exists():
        return model_path

    available_models = sorted(path.name for path in (project_root / "model").glob("*.pt"))
    raise FileNotFoundError(
        f"Modell nicht gefunden: {model_path}. Verfuegbare Modelle in model/: {available_models}"
    )


def main(
    name: str = "yolo26n_bee_beetle_butterfly",
    preset: str = "default",
    model_name: str = "yolo26n-seg.pt",
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"

    if preset not in TRAINING_PRESETS:
        raise ValueError(f"Unbekanntes Trainingspreset: {preset}. Verfuegbar: {sorted(TRAINING_PRESETS)}")
    train_args = TRAINING_PRESETS[preset]

    model_path = _resolve_model_path(project_root, model_name)
    print("starting from model:", model_path)
    model = YOLO(str(model_path))

    data_yaml = _build_output_data_yaml(project_root)

    model.train(
        data=str(data_yaml),
        imgsz=640,
        device="0",
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        **train_args,
    )


    model.val(
        data=str(data_yaml),
        project=str(runs_dir),
        name=f"{name}_val",
        exist_ok=True,
    )
    model.save(str(project_root / "model" / f"{name}-seg.pt"))


if __name__ == "__main__":
    #main(name="yolo26n_bee_beetle_butterfly")
    #main(name="yolo26n_jutestripe_yellowpaper")
    main(
        name="yolo26n_soilspot_real50",
        preset="small_finetune",
        model_name="yolo26n_soilspot-seg.pt",
    )
