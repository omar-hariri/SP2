import wandb
import joblib
from pathlib import Path

from src.models.rf import build_rf_model
from src.training.trainer_utils import (
    load_config,
    load_fold_data,
    evaluate_fold_metrics,
    compute_window_summary,
)
from sklearn.preprocessing import StandardScaler
def run(config: dict):
    """
    Random Forest Training Pipeline.
    Loads data -> Trains -> Evaluates -> Logs to W&B.
    """
    split_config_path = config.get("split_config", "configs/training/data_split.yaml")
    windows_config_path = config.get("windows_config", "configs/training/windows.yaml")
    model_config_path = config.get("model_config", "configs/models/rf.yaml")

    split_cfg   = load_config(split_config_path)
    windows_cfg = load_config(windows_config_path)
    model_cfg   = load_config(model_config_path)
    window_filter = config.get("window", None)

    wandb_cfg    = model_cfg["wandb"]
    ready_dir    = Path(split_cfg["ready_dir"])
    windows      = windows_cfg["windows"]
    folds        = split_cfg["folds"]
    model_params = model_cfg["model"]
    class_names  = model_cfg["output"]["class_names"]

    if window_filter:
        windows = [w for w in windows if w["name"] == window_filter]

    print("=" * 60)
    print("Training Random Forest with Windows (Pipeline)")
    print(f"W&B Project : {wandb_cfg.get('project', 'driver-monitoring-system')}")
    print(f"Windows     : {[w['name'] for w in windows]}")
    print("=" * 60)

    for window_cfg in windows:
        window_name = window_cfg["name"]

        run_wandb = wandb.init(
            project=wandb_cfg.get("project", "driver-monitoring-system"),
            name=f"RF_{window_name}",
            group="Sensor_Benchmarks",
            entity=wandb_cfg.get("entity", None),
            config={**model_params, "window": window_name},
            reinit=True,
        )

        print(f"\n[Window: {window_name}]  W&B Run: {run_wandb.id}")

        fold_results = []

        for fold_cfg in folds:
            fold_num    = fold_cfg["fold"]
            test_driver = fold_cfg["test"]
            fold_dir    = ready_dir / window_name / f"fold{fold_num}"

            if not fold_dir.exists():
                print(f"  SKIP: fold{fold_num} directory not found.")
                continue

            print(f"  Fold {fold_num} (Test: {test_driver}) ...", end=" ", flush=True)

            # 1. Load Data
            X_train, y_train, X_test, y_test = load_fold_data(fold_dir, data_type="ml")

            # 2. Build and Train 
            model = build_rf_model(model_params)
            model.fit(X_train, y_train)

            # 3. Evaluate
            y_pred = model.predict(X_test)
            metrics = evaluate_fold_metrics(y_test, y_pred, class_names, fold_num, log_to_wandb=True)
            fold_results.append(metrics)

            # Save model locally
            models_dir = Path("artifacts/models/rf")
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, models_dir / f"model_{window_name}_fold{fold_num}.joblib")

            print(f"Done.  Acc: {metrics['acc']:.4f}  F1: {metrics['f1']:.4f}")

        # 4. Summarize Window
        compute_window_summary(fold_results, window_name, log_to_wandb=True)
        run_wandb.finish()


