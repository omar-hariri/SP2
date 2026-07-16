# src/utils/metrics.py
# ==================================================
# Utilities for extracting validation and training metrics.

METRIC_KEYS = [
    "epoch",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "val/precision", "val/recall", "val/mAP50", "val/mAP50-95",
]


def extract_epoch_metrics(trainer) -> dict:
    loss = trainer.loss_items
    metrics = trainer.metrics or {}

    return {
        "epoch": trainer.epoch + 1,
        "train/box_loss": float(loss[0]) if loss is not None and len(loss) > 0 else 0.0,
        "train/cls_loss": float(loss[1]) if loss is not None and len(loss) > 1 else 0.0,
        "train/dfl_loss": float(loss[2]) if loss is not None and len(loss) > 2 else 0.0,
        "val/box_loss": float(metrics.get("val/box_loss", 0)),
        "val/cls_loss": float(metrics.get("val/cls_loss", 0)),
        "val/dfl_loss": float(metrics.get("val/dfl_loss", 0)),
        "val/precision": float(metrics.get("metrics/precision(B)", 0)),
        "val/recall": float(metrics.get("metrics/recall(B)", 0)),
        "val/mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
        "val/mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
    }
