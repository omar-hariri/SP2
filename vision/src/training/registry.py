import json
from pathlib import Path


def save_best_weights(weights_path, run_name, registry_path="artifacts/registry.json"):
    registry_file = Path(registry_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    registry = {}
    if registry_file.exists():
        registry = json.loads(registry_file.read_text(encoding="utf-8"))

    registry[run_name] = str(weights_path)
    registry["latest"] = str(weights_path)

    registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def load_latest_weights(registry_path="artifacts/registry.json") -> str:
    registry_file = Path(registry_path)
    if not registry_file.exists():
        raise FileNotFoundError(f"Registry not found: {registry_file}")
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    return registry["latest"]
