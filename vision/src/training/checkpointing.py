from pathlib import Path
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


def mark_run_complete(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / COMPLETE_MARKER
    marker.write_text("complete\n", encoding="utf-8")
    return marker
