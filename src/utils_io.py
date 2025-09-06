# src/utils_io.py
from pathlib import Path
import os, sys, logging, yaml

CONFIG_PATH = os.getenv("CONFIG_PATH", "/opt/airflow/configs/config.yaml")

def get_logger(name="nyc"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    return logging.getLogger(name)

def read_yaml():
    p = Path(CONFIG_PATH)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
