import hashlib
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def train_sha256():
    """Hash of the original train.csv, which MLflow runs must log as data_sha256."""
    path = ROOT / load_params()["data"]["train"]
    return hashlib.sha256(path.read_bytes()).hexdigest()