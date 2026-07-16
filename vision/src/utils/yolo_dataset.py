from pathlib import Path
from typing import Optional, Union

import yaml


DEFAULT_DATASET_DIR = Path("data/processed")
SPLIT_IMAGE_DIRS = {
    "train": Path("train/images"),
    "val": Path("val/images"),
    "test": Path("test/images"),
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def normalize_classes(classes_cfg: dict) -> dict[int, str]:
    classes = classes_cfg.get("classes", classes_cfg)
    return {int(class_id): name for class_id, name in sorted(classes.items(), key=lambda item: int(item[0]))}


def write_yolo_data_yaml(
    dataset_dir: Union[Path, str],
    classes_cfg: dict,
    data_yaml_path: Optional[Union[Path, str]] = None,
) -> Path:
    dataset_dir = Path(dataset_dir)
    data_yaml_path = Path(data_yaml_path) if data_yaml_path else dataset_dir / "data.yaml"
    names = normalize_classes(classes_cfg)

    data = {
        "path": str(dataset_dir.resolve()),
        "train": str(SPLIT_IMAGE_DIRS["train"]).replace("\\", "/"),
        "val": str(SPLIT_IMAGE_DIRS["val"]).replace("\\", "/"),
        "test": str(SPLIT_IMAGE_DIRS["test"]).replace("\\", "/"),
        "nc": len(names),
        "names": names,
    }

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml_path


def missing_dataset_dirs(dataset_dir: Union[Path, str]) -> list:
    dataset_dir = Path(dataset_dir)
    missing = []
    for split_dir in SPLIT_IMAGE_DIRS.values():
        image_dir = dataset_dir / split_dir
        label_dir = dataset_dir / split_dir.parent / "labels"
        if not image_dir.exists():
            missing.append(image_dir)
        if not label_dir.exists():
            missing.append(label_dir)
    return missing


def count_dataset_images(dataset_dir: Union[Path, str]) -> dict:
    dataset_dir = Path(dataset_dir)
    counts = {}
    for split, split_dir in SPLIT_IMAGE_DIRS.items():
        image_dir = dataset_dir / split_dir
        if not image_dir.exists():
            counts[split] = 0
            continue
        counts[split] = sum(
            1
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts
