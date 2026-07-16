# vision/src/models/yolo.py
from typing import Any
from typing import Optional


def load_yolo_model(model_type: str, weights_path: Optional[str] = None) -> Any:
    model_type = model_type.lower().strip()

    if weights_path:
        model_to_load = weights_path
    else:
        if model_type == "yolov8s":
            model_to_load = "yolov8s.pt"
        elif model_type in {"yolo11s", "yolov11s"}:
            model_to_load = "yolo11s.pt"
        else:
            raise ValueError("Unsupported model type. Choose 'yolov8s' or 'yolo11s'.")

    from ultralytics import YOLO

    print(f"Loading model: {model_type.upper()} from {model_to_load}")
    return YOLO(model_to_load)
