# vision/src/pipelines/yolo_pipeline.py
import logging
import re
from pathlib import Path
from typing import Optional, Union

from src.models.yolo import load_yolo_model
from src.training.logging_utils import init_wandb, log_epoch_metrics, log_comparison_plot, print_epoch_summary
from src.training.metrics import extract_epoch_metrics
from src.training.plotting import generate_all_plots
from src.training.registry import save_best_weights
from src.utils.yolo_dataset import count_dataset_images

log = logging.getLogger(__name__)


def _run_name(config: dict, model_type: str) -> str:
    return config.get("wandb", {}).get("experiment_name") or f"{model_type}_run"


def _next_versioned_run_name(base_name: str, runs_dir: Path) -> str:
    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
    versions = []

    if runs_dir.exists():
        for path in runs_dir.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))

    return f"{base_name}_v{max(versions, default=0) + 1}"


def _training_args(config: dict, data_yaml_path: Path, runs_dir: Path, run_name: str) -> dict:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    args = {
        "data": str(data_yaml_path.resolve()),
        "epochs": training_cfg.get("epochs", 50),
        "imgsz": model_cfg.get("img_size", training_cfg.get("imgsz", 640)),
        "batch": training_cfg.get("batch_size", training_cfg.get("batch", 16)),
        "workers": training_cfg.get("workers", 8),
        "project": str(runs_dir),
        "name": run_name,
        "exist_ok": True,
    }

    for key in (
        "device",
        "seed",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "patience",
        "save_period",
        "optimizer",
    ):
        if key in training_cfg:
            args[key] = training_cfg[key]

    args.update(config.get("augmentation", {}))
    return args


def _make_reduce_lr_callback(config: dict):
    cfg = config.get("reduce_lr", {})
    if not cfg.get("enabled", False):
        return None

    monitor = cfg.get("monitor", "val/mAP50-95")
    mode = cfg.get("mode", "max")
    patience = int(cfg.get("patience", 5))
    factor = float(cfg.get("factor", 0.5))
    min_lr = float(cfg.get("min_lr", 1e-6))
    min_delta = float(cfg.get("min_delta", 1e-4))
    cooldown = int(cfg.get("cooldown", 0))

    state = {
        "best": None,
        "bad_epochs": 0,
        "cooldown_left": 0,
    }

    def improved(current: float) -> bool:
        if state["best"] is None:
            return True
        if mode == "min":
            return current < state["best"] - min_delta
        return current > state["best"] + min_delta

    def scale_lr(trainer) -> list[float]:
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is None:
            return []

        next_lrs = []
        for group in optimizer.param_groups:
            current_lr = float(group.get("lr", 0.0))
            next_lr = max(current_lr * factor, min_lr)
            group["lr"] = next_lr
            group["initial_lr"] = max(float(group.get("initial_lr", current_lr)) * factor, min_lr)
            next_lrs.append(next_lr)

        scheduler = getattr(trainer, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [max(float(lr) * factor, min_lr) for lr in scheduler.base_lrs]

        return next_lrs

    def on_epoch_end(trainer, epoch_metrics: dict) -> None:
        current = epoch_metrics.get(monitor)
        if current is None:
            log.warning("ReduceLR monitor '%s' not found in epoch metrics", monitor)
            return

        current = float(current)
        if improved(current):
            state["best"] = current
            state["bad_epochs"] = 0
            return

        if state["cooldown_left"] > 0:
            state["cooldown_left"] -= 1
            return

        state["bad_epochs"] += 1
        if state["bad_epochs"] <= patience:
            return

        next_lrs = scale_lr(trainer)
        state["bad_epochs"] = 0
        state["cooldown_left"] = cooldown
        if next_lrs:
            log.info(
                "ReduceLR: %s did not improve from %.5f; scaled learning rate(s) to %s",
                monitor,
                state["best"],
                ", ".join(f"{lr:.6g}" for lr in next_lrs),
            )

    return on_epoch_end


def _export_training_weights(save_dir: Path, imgsz: int) -> dict:
    exported = {}
    weights_dir = save_dir / "weights"
    for label in ("best", "last"):
        weights_path = weights_dir / f"{label}.pt"
        if not weights_path.exists():
            log.warning("Skipping ONNX export; weights not found: %s", weights_path)
            continue
        try:
            exported[label] = export_yolo_model(str(weights_path), imgsz=imgsz)
        except Exception:
            log.exception("Failed to export %s to ONNX", weights_path)
    return exported


def train_yolo_model(
    model_type: str,
    config: dict,
    data_yaml_path: Union[str, Path],
    runs_dir: Union[str, Path] = "artifacts/runs",
    plots_dir: Optional[Union[str, Path]] = None,
    registry_path: Union[str, Path] = "artifacts/registry.json",
    dataset_dir: Optional[Union[str, Path]] = None,
) -> dict:
    data_yaml_path = Path(data_yaml_path)
    runs_dir = Path(runs_dir)
    base_run_name = _run_name(config, model_type)
    run_name = _next_versioned_run_name(base_run_name, runs_dir)
    weights_path = config.get("model", {}).get("weights")
    model = load_yolo_model(model_type, weights_path)

    counts = count_dataset_images(dataset_dir or data_yaml_path.parent)
    init_wandb(config, run_name, counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))

    history = {
        "epoch": [],
        "train/box_loss": [], "train/cls_loss": [], "train/dfl_loss": [],
        "val/box_loss": [], "val/cls_loss": [], "val/dfl_loss": [],
        "val/precision": [], "val/recall": [], "val/mAP50": [], "val/mAP50-95": [],
    }
    total_epochs = config.get("training", {}).get("epochs", 50)
    reduce_lr_on_epoch_end = _make_reduce_lr_callback(config)

    def on_fit_epoch_end(trainer):
        epoch_metrics = extract_epoch_metrics(trainer)

        for key in history:
            if key in epoch_metrics:
                history[key].append(epoch_metrics[key])

        if reduce_lr_on_epoch_end:
            reduce_lr_on_epoch_end(trainer, epoch_metrics)

        log_epoch_metrics(epoch_metrics)
        print_epoch_summary(epoch_metrics, total_epochs)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(**_training_args(config, data_yaml_path, runs_dir, run_name))
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    exported_onnx = _export_training_weights(
        save_dir,
        imgsz=model_cfg.get("img_size", training_cfg.get("imgsz", 640)),
    )

    save_best_weights(best_weights, run_name, registry_path=registry_path)
    plots_dir = Path(plots_dir) if plots_dir else save_dir
    generate_all_plots(history, plots_dir, log_fn=log_comparison_plot)

    log.info("Training completed successfully for %s", model_type.upper())
    log.info("Best weights registered at: %s", best_weights)
    if exported_onnx:
        log.info("ONNX exports: %s", exported_onnx)

    return {
        "best_weights": str(best_weights),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "best_onnx": exported_onnx.get("best"),
        "last_onnx": exported_onnx.get("last"),
        "results_dir": str(save_dir),
    }


def run_training(
    model_cfg: dict,
    data_yaml_path: Union[str, Path],
    dataset_dir: Optional[Union[str, Path]] = None,
    runs_dir: Union[str, Path] = "artifacts/runs",
    plots_dir: Optional[Union[str, Path]] = None,
) -> str:
    model_type = model_cfg.get("model", {}).get("name", "yolov8s").lower()
    result = train_yolo_model(
        model_type=model_type,
        config=model_cfg,
        data_yaml_path=data_yaml_path,
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        plots_dir=plots_dir,
    )
    return result["best_weights"]


def export_yolo_model(weights_path: str, imgsz: int = 640) -> str:
    from ultralytics import YOLO

    log.info("Exporting PyTorch weights %s to ONNX format", weights_path)
    model = YOLO(weights_path)

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,
        dynamic=False,
        simplify=True
    )

    log.info("ONNX export completed. File saved at: %s", onnx_path)
    return onnx_path
