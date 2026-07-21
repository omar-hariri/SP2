import wandb
import numpy as np
from pathlib import Path

from src.models.lstm import build_lstm_model
from src.training.trainer_utils import (
load_config,
load_fold_data,
evaluate_fold_metrics,
compute_window_summary,
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from wandb.integration.keras import WandbMetricsLogger

def run(config: dict):
    """
    LSTM Training Pipeline.
    Loads data -> Trains -> Evaluates -> Logs to W&B.
    """
    split_config_path = config.get("split_config", "configs/training/data_split.yaml")
    windows_config_path = config.get("windows_config", "configs/training/windows.yaml")
    lstm_config_path = config.get("model_config", "configs/models/lstm.yaml")

    split_cfg   = load_config(split_config_path)
    windows_cfg = load_config(windows_config_path)
    lstm_cfg    = load_config(lstm_config_path)
    window_filter = config.get("window", None)

    ready_dir    = Path(split_cfg["ready_dir"])
    windows      = windows_cfg["windows"]
    folds        = split_cfg["folds"]
    class_names  = lstm_cfg["data"]["classes"]
    model_params = lstm_cfg["model"]
    train_params = lstm_cfg["training"]

    if window_filter:
        windows = [w for w in windows if w["name"] == window_filter]
        if not windows:
            print(f"ERROR: window '{window_filter}' not found in windows config.")
            return

    print("=" * 60)
    print("Training LSTM with Windows (Pipeline)")
    print(f"W&B Project : {lstm_cfg['wandb']['project']}")
    print(f"Windows     : {[w['name'] for w in windows]}")
    print("=" * 60)

    for window_cfg in windows:
        window_name = window_cfg["name"]

        print(f"\n[Window: {window_name}]")

        fold_results = []

        for fold_cfg in folds:
            fold_num    = fold_cfg["fold"]
            test_driver = fold_cfg["test"]
            fold_dir    = ready_dir / window_name / f"fold{fold_num}"

            if not fold_dir.exists():
                print(f"  SKIP: fold{fold_num} directory not found.")
                continue

            print(f"  Fold {fold_num} (Test: {test_driver}) ...", end=" ", flush=True)

            # Start a W&B run for this fold (grouped by window)
            run_wandb = wandb.init(
                project=lstm_cfg["wandb"]["project"],
                name=f"LSTM_{window_name}_fold{fold_num}",
                group=window_name,
                config={**model_params, **train_params, "window": window_name, "fold": fold_num},
                reinit=True,
            )

            print(f"W&B Run: {run_wandb.id}")

            # 1. Load Data
            X_train, y_train, X_test, y_test = load_fold_data(fold_dir, data_type="lstm")

            # 2. Build and Train
            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                cfg={
                    **model_params,
                    "lr": train_params["learning_rate"],
                    "num_classes": len(class_names),
                },
            )

            callbacks = [
                EarlyStopping(
                    monitor=train_params.get("monitor", "val_loss"),
                    patience=train_params["early_stopping_patience"],
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor=train_params.get("monitor", "val_loss"),
                    patience=train_params["reduction_patience"],
                    factor=0.5,
                ),
                WandbMetricsLogger(),
            ]
            from sklearn.utils.class_weight import compute_class_weight

            classes = np.unique(y_train)

            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train
            )
            class_weight_dict = {int(c): w for c, w in zip(classes, weights)}

            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=train_params["epochs"],
                batch_size=train_params["batch_size"],
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1,
            )

            # 3. Evaluate
            y_probs = model.predict(X_test)
            y_pred = (y_probs > 0.5).astype(int)

            metrics = evaluate_fold_metrics(y_test, y_pred, class_names, fold_num, log_to_wandb=True)
            fold_results.append(metrics)

            # Save model locally Keras
            models_dir = Path("artifacts/models/lstm")
            models_dir.mkdir(parents=True, exist_ok=True)
            model.save(models_dir / f"model_{window_name}_fold{fold_num}.keras")

            print(f"Done.  Acc: {metrics['acc']:.4f}  F1: {metrics['f1']:.4f}")
            # Finish this fold's run
            run_wandb.finish()

        # 4. Summarize Window: create a summary run grouped with the folds
        run_summary = wandb.init(
            project=lstm_cfg["wandb"]["project"],
            name=f"LSTM_{window_name}_summary",
            group=window_name,
            config={"window": window_name},
            reinit=True,
        )

        compute_window_summary(fold_results, window_name, log_to_wandb=True)
        run_summary.finish()

