from pathlib import Path
from shutil import copy2, copytree


def mirror_file(source_path: Path, target_path: Path):
    source_path = Path(source_path)
    target_path = Path(target_path)
    if not source_path.exists():
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(source_path, target_path)
    return target_path


def mirror_directory(source_dir: Path, target_dir: Path):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.exists():
        return None

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    copytree(source_dir, target_dir, dirs_exist_ok=True)
    return target_dir


def mirror_epoch_checkpoint(source_run_dir: Path, mirror_runs_dir: Path | None, epoch: int):
    if mirror_runs_dir is None:
        return None

    source_run_dir = Path(source_run_dir)
    mirror_runs_dir = Path(mirror_runs_dir)
    source_weights = source_run_dir / "weights" / "last.pt"
    if not source_weights.exists():
        return None

    target_run_dir = mirror_runs_dir / source_run_dir.name
    target_weights_dir = target_run_dir / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)

    copy2(source_weights, target_weights_dir / "last.pt")

    target_epoch_dir = target_weights_dir / "epochs"
    target_epoch_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_epoch_dir / f"epoch_{epoch:04d}.pt"
    copy2(source_weights, target_checkpoint)
    return target_checkpoint


def mirror_run_artifacts(
    source_run_dir: Path,
    mirror_runs_dir: Path | None = None,
    source_plots_dir: Path | None = None,
    mirror_plots_dir: Path | None = None,
    source_registry_path: Path | None = None,
    mirror_registry_path: Path | None = None,
):
    mirrored = {}

    if mirror_runs_dir is not None:
        target_run_dir = Path(mirror_runs_dir) / Path(source_run_dir).name
        mirrored["run_dir"] = mirror_directory(source_run_dir, target_run_dir)

    if source_plots_dir is not None and mirror_plots_dir is not None:
        mirrored["plots_dir"] = mirror_directory(source_plots_dir, mirror_plots_dir)

    if source_registry_path is not None and mirror_registry_path is not None:
        mirrored["registry_path"] = mirror_file(source_registry_path, mirror_registry_path)

    return mirrored