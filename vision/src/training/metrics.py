# src/utils/metrics.py
# ==================================================
# Utilities for extracting validation and training metrics.


def _get_loss(loss, index: int, key: str) -> float:
    if loss is None:
        return 0.0
    if isinstance(loss, dict):
        return float(loss.get(key, 0))
    try:
        return float(loss[index]) if len(loss) > index else 0.0
    except (IndexError, KeyError, TypeError):
        return 0.0


def extract_epoch_metrics(trainer) -> dict:
    loss = trainer.loss_items
    metrics = trainer.metrics or {}

    return {
        "epoch": trainer.epoch + 1,
        "train/box_loss": _get_loss(loss, 0, "box_loss"),
        "train/cls_loss": _get_loss(loss, 1, "cls_loss"),
        "train/dfl_loss": _get_loss(loss, 2, "dfl_loss"),
        "val/box_loss": float(metrics.get("val/box_loss", 0)),
        "val/cls_loss": float(metrics.get("val/cls_loss", 0)),
        "val/dfl_loss": float(metrics.get("val/dfl_loss", 0)),
        "val/precision": float(metrics.get("metrics/precision(B)", 0)),
        "val/recall": float(metrics.get("metrics/recall(B)", 0)),
        "val/mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
        "val/mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
    }
