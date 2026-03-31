from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SOURCE = PROJECT_ROOT / "datasets" / "creditcard.csv"

DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_directories() -> None:
    for directory in (
        DATA_DIR,
        DATA_DIR / "raw",
        SAMPLES_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        TABLES_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
