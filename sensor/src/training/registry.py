# src/training/registry.py
# ========================
# Uses lazy imports to prevent loading unnecessary dependencies (e.g. torch, tensorflow)
# when only one pipeline is being executed.

def run_rf_pipeline(config):
    from src.pipelines.rf_pipeline import run
    return run(config)

def run_xgb_pipeline(config):
    from src.pipelines.xgb_pipeline import run
    return run(config)

def run_lstm_pipeline(config):
    from src.pipelines.lstm_pipeline import run
    return run(config)

PIPELINES = {
    "rf":   run_rf_pipeline,
    "xgb":  run_xgb_pipeline,
    "lstm": run_lstm_pipeline,
}

