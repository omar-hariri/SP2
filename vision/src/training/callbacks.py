import logging
from typing import Callable, MutableMapping, Optional

from src.training.logging_utils import log_epoch_metrics, print_epoch_summary
from src.training.metrics import extract_epoch_metrics


log = logging.getLogger(__name__)


def build_history() -> dict:
    return {
        "epoch": [],
        "train/box_loss": [],
        "train/cls_loss": [],
        "train/dfl_loss": [],
        "val/box_loss": [],
        "val/cls_loss": [],
        "val/dfl_loss": [],
        "val/precision": [],
        "val/recall": [],
        "val/mAP50": [],
        "val/mAP50-95": [],
    }


def build_reduce_lr_callback(config: dict):
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


def build_epoch_end_callback(
    history: MutableMapping[str, list],
    total_epochs: int,
    reduce_lr_callback: Optional[Callable] = None,
    checkpoint_callback: Optional[Callable] = None,
):
    def on_fit_epoch_end(trainer):
        epoch_metrics = extract_epoch_metrics(trainer)

        for key in history:
            if key in epoch_metrics:
                history[key].append(epoch_metrics[key])

        if reduce_lr_callback:
            reduce_lr_callback(trainer, epoch_metrics)

        if checkpoint_callback:
            checkpoint_callback(trainer, epoch_metrics)

        log_epoch_metrics(epoch_metrics)
        print_epoch_summary(epoch_metrics, total_epochs)

    return on_fit_epoch_end
