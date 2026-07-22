import re
from pathlib import Path


COMPLETE_MARKER = ".complete"


def build_run_name(config: dict, model_type: str) -> str:
    return config.get("wandb", {}).get("experiment_name") or f"{model_type}_run"


def next_versioned_run_name(base_name: str, runs_dir: Path) -> str:
    runs_dir = Path(runs_dir)
    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
    latest_version = None
    latest_complete = True

    if runs_dir.exists():
        for path in runs_dir.iterdir():
            if not path.is_dir():
                continue

            match = pattern.match(path.name)
            if match is None:
                continue

            version = int(match.group(1))
            if latest_version is None or version > latest_version:
                latest_version = version
                latest_complete = (path / COMPLETE_MARKER).exists()

    if latest_version is None:
        return f"{base_name}_v1"

    if latest_complete:
        return f"{base_name}_v{latest_version + 1}"

    return f"{base_name}_v{latest_version}"


def prepare_run(config: dict, model_type: str, runs_dir: Path) -> dict:
    base_name = build_run_name(config, model_type)
    run_name = next_versioned_run_name(base_name, runs_dir)
    run_dir = Path(runs_dir) / run_name
    resume_checkpoint = run_dir / "weights" / "last.pt"
    if not resume_checkpoint.exists():
        resume_checkpoint = None

    return {
        "base_name": base_name,
        "run_name": run_name,
        "run_dir": run_dir,
        "resume_checkpoint": resume_checkpoint,
    }


WANDB_ID_FILE = ".wandb_run_id"


def save_wandb_run_id(run_dir: Path, run_id: str):
    path = Path(run_dir) / WANDB_ID_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id.strip(), encoding="utf-8")


def load_wandb_run_id(run_dir: Path) -> str | None:
    path = Path(run_dir) / WANDB_ID_FILE
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def mark_run_complete(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / COMPLETE_MARKER
    marker.write_text("complete\n", encoding="utf-8")
    return marker
