import argparse
import sys
from src.training.registry import PIPELINES

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Entry Point for Sensor Branch")
    parser.add_argument("--pipeline", type=str, required=True, choices=list(PIPELINES.keys()),
                        help="Which pipeline to execute")

    parser.add_argument("--split_config", type=str, default="configs/training/data_split.yaml",
                        help="Path to data split configuration")
    parser.add_argument("--windows_config", type=str, default="configs/training/windows.yaml",
                        help="Path to windows configuration")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Model specific configuration path. Defaults to configs/models/<pipeline>.yaml if not provided")
    parser.add_argument("--window", type=str, default=None,
                        help="Train on a single window (e.g. w6s). Omit to train all.")

    return parser.parse_args()

def main():
    args = parse_args()

    pipeline_fn = PIPELINES[args.pipeline]

    model_config = args.model_config
    if not model_config:
        model_config = f"configs/models/{args.pipeline}.yaml"

    config = {
        "split_config": args.split_config,
        "windows_config": args.windows_config,
        "model_config": model_config,
        "window": args.window
    }

    print(f"Executing pipeline: {args.pipeline}")
    pipeline_fn(config)

if __name__ == "__main__":
    main()