import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)
_wandb: Optional[Any] = None
_wandb_run: Optional[Any] = None
_wandb_warning_shown = False


def _load_wandb() -> Optional[Any]:
    global _wandb, _wandb_warning_shown
    if _wandb is not None:
        return _wandb
    try:
        _wandb = import_module("wandb")
    except ImportError:
        if not _wandb_warning_shown:
            log.warning("wandb is not installed; training will continue without WandB logging")
            _wandb_warning_shown = True
        return None
    return _wandb


def init_wandb(cfg: dict, run_name: str, n_train: int, n_val: int, n_test: int):
    global _wandb_run
    wcfg = cfg.get("wandb", {})
    if not wcfg or not wcfg.get("enabled", True):
        log.info("WandB logging is disabled")
        return None

    wandb = _load_wandb()
    if wandb is None:
        return None

    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    run = wandb.init(
        project=wcfg["project"],
        entity=wcfg.get("entity"),
        name=run_name,
        config={
            **model_cfg,
            **training_cfg,
            "augmentation": cfg.get("augmentation", {}),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        },
        tags=wcfg.get("tags", ["vision", "driver-monitoring"]),
    )
    _wandb_run = run
    log.info(f"WandB run initialized -> {run.url}")
    return run


def log_eda_images(plots_dir):
    wandb = _load_wandb()
    if wandb is None or _wandb_run is None:
        return
    plots_dir = Path(plots_dir)
    for plot_file, key in [
        ("eda_distribution.png", "eda/class_distribution"),
        ("split_distribution.png", "eda/split_distribution"),
        ("sample_annotations.png", "eda/sample_annotations"),
    ]:
        p = plots_dir / plot_file
        if p.exists():
            wandb.log({key: wandb.Image(str(p))})


def log_epoch_metrics(epoch_metrics: dict):
    wandb = _load_wandb()
    if wandb is None or _wandb_run is None:
        return
    wandb.log(epoch_metrics, step=epoch_metrics["epoch"])


def log_comparison_plot(key: str, path):
    wandb = _load_wandb()
    if wandb is None or _wandb_run is None:
        return
    wandb.log({key: wandb.Image(str(path))})


def print_epoch_summary(epoch_metrics: dict, total_epochs: int):
    sep = "-" * 65
    log.info(sep)
    log.info(f"  Epoch {epoch_metrics['epoch']}/{total_epochs}")
    log.info(sep)
    log.info("  LOSS")
    log.info(f"     Train -> Box={epoch_metrics['train/box_loss']:.4f}  "
              f"Cls={epoch_metrics['train/cls_loss']:.4f}  Dfl={epoch_metrics['train/dfl_loss']:.4f}")
    log.info(f"     Val   -> Box={epoch_metrics['val/box_loss']:.4f}  "
              f"Cls={epoch_metrics['val/cls_loss']:.4f}  Dfl={epoch_metrics['val/dfl_loss']:.4f}")
    log.info("  VAL METRICS")
    log.info(f"     Precision={epoch_metrics['val/precision']:.4f}  Recall={epoch_metrics['val/recall']:.4f}")
    log.info(f"     mAP@0.5={epoch_metrics['val/mAP50']:.4f}  mAP@0.95={epoch_metrics['val/mAP50-95']:.4f}")
    log.info(sep)
