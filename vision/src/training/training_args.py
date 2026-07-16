from pathlib import Path


def build_training_args(
    config: dict,
    data_yaml_path: Path,
    runs_dir: Path,
    run_name: str,
    resume: bool = False,
) -> dict:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    args = {
        "data": str(Path(data_yaml_path).resolve()),
        "epochs": training_cfg.get("epochs", 50),
        "imgsz": model_cfg.get("img_size", training_cfg.get("imgsz", 640)),
        "batch": training_cfg.get("batch_size", training_cfg.get("batch", 16)),
        "workers": training_cfg.get("workers", 8),
        "project": str(Path(runs_dir)),
        "name": run_name,
        "exist_ok": True,
        "save_period": 1,
    }

    if resume:
        args["resume"] = True

    for key in (
        "device",
        "seed",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "patience",
        "optimizer",
    ):
        if key in training_cfg:
            args[key] = training_cfg[key]

    args.update(config.get("augmentation", {}))
    return args
