import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

LOSS_PAIRS = [
    ("Box Loss", "train/box_loss", "val/box_loss"),
    ("Cls Loss", "train/cls_loss", "val/cls_loss"),
    ("Dfl Loss", "train/dfl_loss", "val/dfl_loss"),
]

METRIC_KEYS = [
    ("Precision", "val/precision"),
    ("Recall", "val/recall"),
    ("mAP@0.5", "val/mAP50"),
    ("mAP@0.95", "val/mAP50-95"),
]


def plot_losses(history: dict, out_dir: Path) -> Path:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (title, train_key, val_key) in zip(axes, LOSS_PAIRS):
        ax.plot(epochs, history[train_key], label="Train")
        ax.plot(epochs, history[val_key], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    path = out_dir / "losses.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_metrics(history: dict, out_dir: Path) -> Path:
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, key in METRIC_KEYS:
        ax.plot(epochs, history[key], label=label)

    ax.set_title("Validation Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    path = out_dir / "metrics.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def generate_all_plots(history: dict, out_dir: Path, log_fn=None):
    if not history["epoch"]:
        log.warning("History is empty, skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("comparison/losses", plot_losses(history, out_dir)),
        ("comparison/metrics", plot_metrics(history, out_dir)),
    ]
    for key, path in plots:
        log.info(f"Plot saved -> {path}")
        if log_fn:
            log_fn(key, path)
