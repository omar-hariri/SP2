import logging
from pathlib import Path
from shutil import copy2


log = logging.getLogger(__name__)


def export_yolo_model(weights_path: str, imgsz: int = 640) -> str:
    from ultralytics import YOLO  # type: ignore[import-not-found]

    log.info("Exporting PyTorch weights %s to ONNX format", weights_path)
    model = YOLO(weights_path)

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,
        dynamic=False,
        simplify=True,
    )

    log.info("ONNX export completed. File saved at: %s", onnx_path)
    return onnx_path


def export_training_weights(save_dir: Path, imgsz: int) -> dict:
    exported = {}
    weights_dir = Path(save_dir) / "weights"

    for label in ("best", "last"):
        weights_path = weights_dir / f"{label}.pt"
        if not weights_path.exists():
            log.warning("Skipping ONNX export; weights not found: %s", weights_path)
            continue

        try:
            exported_path = Path(export_yolo_model(str(weights_path), imgsz=imgsz))
            if exported_path.exists() and exported_path != weights_dir / exported_path.name:
                copy2(exported_path, weights_dir / exported_path.name)
                exported[label] = str(weights_dir / exported_path.name)
            else:
                exported[label] = str(exported_path)
        except Exception:
            log.exception("Failed to export %s to ONNX", weights_path)

    return exported
