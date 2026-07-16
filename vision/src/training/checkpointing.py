from pathlib import Path
from shutil import copy2


COMPLETE_MARKER = ".complete"


def resolve_resume_checkpoint(run_dir: Path):
    run_dir = Path(run_dir)
    checkpoint = run_dir / "weights" / "last.pt"
    if checkpoint.exists() and not (run_dir / COMPLETE_MARKER).exists():
        return checkpoint
    return None


def best_and_last_weights(save_dir: Path) -> dict:
    save_dir = Path(save_dir)
    weights_dir = save_dir / "weights"
    return {
        "best": weights_dir / "best.pt",
        "last": weights_dir / "last.pt",
    }


def preserve_epoch_checkpoint(save_dir: Path, epoch: int) -> Path | None:
    if save_dir is None:
        return None

    save_dir = Path(save_dir)
    weights_dir = save_dir / "weights"
    last_weights = weights_dir / "last.pt"
    if not last_weights.exists():
        return None

    epoch_dir = weights_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = epoch_dir / f"epoch_{epoch:04d}.pt"
    copy2(last_weights, checkpoint_path)
    return checkpoint_path


def mark_run_complete(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / COMPLETE_MARKER
    marker.write_text("complete\n", encoding="utf-8")
    return marker
