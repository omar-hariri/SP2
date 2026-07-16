# vision/src/pipelines/yolo_pipeline.py
import logging
from pathlib import Path
from typing import Optional, Union

from src.models.yolo import load_yolo_model
from src.training.artifact_mirroring import mirror_epoch_checkpoint, mirror_run_artifacts
from src.training.callbacks import build_epoch_end_callback, build_history, build_reduce_lr_callback
from src.training.checkpointing import best_and_last_weights, mark_run_complete, preserve_epoch_checkpoint, resolve_resume_checkpoint
from src.training.export import export_training_weights, export_yolo_model as _export_yolo_model
from src.training.logging_utils import init_wandb, log_comparison_plot
from src.training.plotting import generate_all_plots
from src.training.registry import save_best_weights
from src.training.run_management import prepare_run
from src.training.training_args import build_training_args
from src.utils.yolo_dataset import count_dataset_images

log = logging.getLogger(__name__)


def train_yolo_model(
    model_type: str,
    config: dict,
    data_yaml_path: Union[str, Path],
    runs_dir: Union[str, Path] = "artifacts/runs",
    plots_dir: Optional[Union[str, Path]] = None,
    registry_path: Union[str, Path] = "artifacts/registry.json",
    dataset_dir: Optional[Union[str, Path]] = None,
    mirror_to_drive: bool = False,
    drive_runs_dir: Optional[Union[str, Path]] = None,
    drive_plots_dir: Optional[Union[str, Path]] = None,
    drive_registry_path: Optional[Union[str, Path]] = None,
) -> dict:
    data_yaml_path = Path(data_yaml_path)
    runs_dir = Path(runs_dir)
    run_state = prepare_run(config, model_type, runs_dir)
    resume_checkpoint = resolve_resume_checkpoint(run_state["run_dir"])
    weights_path = resume_checkpoint or config.get("model", {}).get("weights")
    model = load_yolo_model(model_type, weights_path)

    counts = count_dataset_images(dataset_dir or data_yaml_path.parent)
    init_wandb(
        config,
        run_state["run_name"],
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("test", 0),
    )

    history = build_history()
    total_epochs = config.get("training", {}).get("epochs", 50)
    reduce_lr_on_epoch_end = build_reduce_lr_callback(config)
    on_fit_epoch_end = build_epoch_end_callback(
        history,
        total_epochs,
        reduce_lr_on_epoch_end,
        checkpoint_callback=lambda trainer, epoch_metrics: preserve_epoch_checkpoint(
            getattr(trainer, "save_dir", None),
            epoch_metrics["epoch"],
        ),
    )

    if mirror_to_drive and drive_runs_dir is not None:
        drive_runs_dir = Path(drive_runs_dir)
        on_fit_epoch_end = build_epoch_end_callback(
            history,
            total_epochs,
            reduce_lr_on_epoch_end,
            checkpoint_callback=lambda trainer, epoch_metrics: (
                preserve_epoch_checkpoint(getattr(trainer, "save_dir", None), epoch_metrics["epoch"]),
                mirror_epoch_checkpoint(Path(trainer.save_dir), drive_runs_dir, epoch_metrics["epoch"]),
            ),
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(
        **build_training_args(
            config,
            data_yaml_path,
            runs_dir,
            run_state["run_name"],
            resume=resume_checkpoint is not None,
        )
    )
    save_dir = Path(results.save_dir)
    weights_paths = best_and_last_weights(save_dir)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    exported_onnx = export_training_weights(
        save_dir,
        imgsz=model_cfg.get("img_size", training_cfg.get("imgsz", 640)),
    )

    save_best_weights(weights_paths["best"], run_state["run_name"], registry_path=registry_path)
    plots_dir = Path(plots_dir) if plots_dir else save_dir
    generate_all_plots(history, plots_dir, log_fn=log_comparison_plot)
    mark_run_complete(save_dir)

    if mirror_to_drive:
        mirror_run_artifacts(
            save_dir,
            mirror_runs_dir=Path(drive_runs_dir) if drive_runs_dir is not None else None,
            source_plots_dir=plots_dir,
            mirror_plots_dir=Path(drive_plots_dir) if drive_plots_dir is not None else None,
            source_registry_path=Path(registry_path),
            mirror_registry_path=Path(drive_registry_path) if drive_registry_path is not None else None,
        )

    log.info("Training completed successfully for %s", model_type.upper())
    log.info("Best weights registered at: %s", weights_paths["best"])
    if exported_onnx:
        log.info("ONNX exports: %s", exported_onnx)

    return {
        "best_weights": str(weights_paths["best"]),
        "last_weights": str(weights_paths["last"]),
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
    mirror_to_drive: bool = False,
    drive_runs_dir: Optional[Union[str, Path]] = None,
    drive_plots_dir: Optional[Union[str, Path]] = None,
    drive_registry_path: Optional[Union[str, Path]] = None,
) -> str:
    model_type = model_cfg.get("model", {}).get("name", "yolov8s").lower()
    result = train_yolo_model(
        model_type=model_type,
        config=model_cfg,
        data_yaml_path=data_yaml_path,
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        plots_dir=plots_dir,
        mirror_to_drive=mirror_to_drive,
        drive_runs_dir=drive_runs_dir,
        drive_plots_dir=drive_plots_dir,
        drive_registry_path=drive_registry_path,
    )
    return result["best_weights"]


def export_yolo_model(weights_path: str, imgsz: int = 640) -> str:
    return _export_yolo_model(weights_path, imgsz=imgsz)
