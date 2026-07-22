# vision_trainer.py
# ==================================================
# Main entry point for training YOLO models.

import argparse
import logging
from pathlib import Path

from src.pipelines.yolo_pipeline import run_training
from src.utils.config_loader import load_config
from src.utils.yolo_dataset import DEFAULT_DATASET_DIR, missing_dataset_dirs, write_yolo_data_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO vision model.")
    parser.add_argument("--model-config", default="configs/models/yolov8s.yaml")
    parser.add_argument("--classes-config", default="configs/data/classes.yaml")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--plots-dir", default="artifacts/plots")
    parser.add_argument("--drive-root", default=None,
                        help="Google Drive root (e.g. /content/drive/MyDrive/yolo). "
                             "When set, runs/plots/registry are mirrored here.")
    parser.add_argument("--data-yaml", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = load_config(args.model_config)
    classes_cfg = load_config(args.classes_config)

    dataset_dir = Path(args.dataset_dir)
    missing_dirs = missing_dataset_dirs(dataset_dir)
    if missing_dirs:
        missing = "\n".join(f"  - {path}" for path in missing_dirs)
        raise FileNotFoundError(
            "Processed YOLO dataset is missing required folders:\n"
            f"{missing}\n"
            "Run src/preprocessing/vision_preprocessor.py before training."
        )

    data_yaml_path = Path(args.data_yaml) if args.data_yaml else dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        data_yaml_path = write_yolo_data_yaml(dataset_dir, classes_cfg, data_yaml_path)
    runs_dir = Path(args.runs_dir)
    plots_dir = Path(args.plots_dir)
    drive_root = Path(args.drive_root) if args.drive_root else None

    best_weights = run_training(
        model_cfg,
        data_yaml_path,
        dataset_dir,
        runs_dir,
        plots_dir,
        drive_root=drive_root,
    )

    log.info(f"Training complete. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
