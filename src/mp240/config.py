from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)