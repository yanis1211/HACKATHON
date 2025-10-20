"""Global configuration for the Flu POC project."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SERVING_DATA_DIR = DATA_DIR / "serving"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


for directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SERVING_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
