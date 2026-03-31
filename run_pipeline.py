from __future__ import annotations

from src.config import ensure_directories
from src.data.preparation import run_stage_3_preparation
from src.data.profile import run_bootstrap
from src.evaluation.stage5 import run_stage_5_evaluation
from src.evaluation.stage5b_statistical_testing import run_stage_5b_statistical_testing
from src.models.stage4 import run_stage_4_modeling
from src.models.stage4b_validation import run_stage_4b_validation


def main() -> None:
    ensure_directories()
    run_bootstrap()
    run_stage_3_preparation()
    run_stage_4_modeling()
    run_stage_4b_validation()  # ← NEW: Repeated stratified CV validation
    run_stage_5_evaluation()
    run_stage_5b_statistical_testing()  # ← NEW: Statistical significance testing
    print("Research pipeline completed. Check results/, data/samples/, data/processed/, and results/models/.")


if __name__ == "__main__":
    main()