import logging
from pathlib import Path
from typing import Optional, Union

from src.models.yolo import load_yolo_model
from src.training.artifact_mirroring import mirror_run_artifacts
from src.training.callbacks import build_batch_loss_accumulator, build_epoch_end_callback, build_history, build_reduce_lr_callback
from src.training.checkpointing import best_and_last_weights, mark_run_complete, resolve_resume_checkpoint
from src.training.export import export_training_weights, export_yolo_model as _export_yolo_model
from src.training.logging_utils import init_wandb, log_comparison_plot
from src.training.plotting import generate_all_plots
from src.training.registry import save_best_weights
from src.training.run_management import load_wandb_run_id, prepare_run, save_wandb_run_id
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
    drive_root: Optional[Union[str, Path]] = None,
) -> dict:
    data_yaml_path = Path(data_yaml_path)
    runs_dir = Path(runs_dir)
    run_state = prepare_run(config, model_type, runs_dir)
    resume_checkpoint = resolve_resume_checkpoint(run_state["run_dir"])
    weights_path = resume_checkpoint or config.get("model", {}).get("weights")
    model = load_yolo_model(model_type, weights_path)

    is_resume = resume_checkpoint is not None
    wandb_run_id = load_wandb_run_id(run_state["run_dir"]) if is_resume else None

    counts = count_dataset_images(dataset_dir or data_yaml_path.parent)
    run = init_wandb(
        config,
        run_state["run_name"],
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("test", 0),
        resume=is_resume,
        run_id=wandb_run_id,
    )

    if run is not None and not is_resume:
        save_wandb_run_id(run_state["run_dir"], run.id)

    history = build_history()
    total_epochs = config.get("training", {}).get("epochs", 50)
    reduce_lr_on_epoch_end = build_reduce_lr_callback(config)
    on_batch_end, loss_flush = build_batch_loss_accumulator()
    on_fit_epoch_end = build_epoch_end_callback(
        history, total_epochs, loss_flush, reduce_lr_on_epoch_end,
    )

    model.add_callback("on_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(
        **build_training_args(
            config,
            data_yaml_path,
            runs_dir,
            run_state["run_name"],
            resume=is_resume,
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

    if drive_root is not None:
        drive_root = Path(drive_root)
        mirror_run_artifacts(
            save_dir,
            mirror_runs_dir=drive_root / "runs",
            source_plots_dir=plots_dir,
            mirror_plots_dir=drive_root / "plots",
            source_registry_path=Path(registry_path),
            mirror_registry_path=drive_root / "registry.json",
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
    drive_root: Optional[Union[str, Path]] = None,
) -> str:
    model_type = model_cfg.get("model", {}).get("name", "yolov8s").lower()
    result = train_yolo_model(
        model_type=model_type,
        config=model_cfg,
        data_yaml_path=data_yaml_path,
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        plots_dir=plots_dir,
        drive_root=drive_root,
    )
    return result["best_weights"]


def export_yolo_model(weights_path: str, imgsz: int = 640) -> str:
    return _export_yolo_model(weights_path, imgsz=imgsz)
