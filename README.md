# Fraud Detection on Waqf Transaction

This repository is now scaffolded as a reproducible research project for the paper:

**Enhancing Trust in Digital Waqf Platforms through Machine Learning-Based Fraud Detection**

## Current scope

- Research workflow follows CRISP-DM.
- The currently available dataset is `datasets/creditcard.csv`.
- Because a real waqf fraud dataset is not available, this project treats credit card fraud as a proxy domain for suspicious digital waqf transaction behavior.
- The current implementation covers:
  - project structure
  - stage 1 research packaging
  - stage 2 dataset profiling and baseline EDA outputs
  - stage 3 data preparation for supervised and unsupervised modeling
  - stage 4 modeling for Random Forest, XGBoost, Isolation Forest, and Autoencoder
  - stage 5 comparative evaluation and waqf-context interpretation

## Project structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── datasets/
├── ideas/
├── notebooks/
├── referensi/
├── references/
├── results/
│   ├── models/
│   ├── plots/
│   ├── reports/
│   └── tables/
├── logs/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── utils/
├── requirements.txt
├── requirements-modeling.txt
└── run_pipeline.py
```

## Quick start

1. Install base dependencies:

```bash
pip install -r requirements.txt
```

2. Run the research pipeline:

```bash
python run_pipeline.py
```

3. Review generated outputs in:

- `results/reports/`
- `results/tables/`
- `results/plots/`
- `data/samples/`
- `data/processed/`

## Notes

- The modeling stack is split from the base stack because some deep learning dependencies can be sensitive to OS and Python version.
- Stage 3 currently removes duplicate rows, keeps all non-target features, performs a stratified train-validation split, computes class weights, and saves transformed datasets for both supervised and anomaly-detection workflows.
- Stage 4 trains four baseline models from the saved Stage 3 datasets, writes per-model reports, saves trained models, and exports evaluation plots and a metrics comparison table.
- Supervised model thresholds are calibrated on validation predictions to align with narrative goals: precision-priority for Random Forest and recall-priority (with precision floor) for XGBoost.
- Stage 5 converts Stage 4 metrics into comparative evaluation outputs, narrative interpretation for digital waqf governance, and explicit validity-limit discussion for the proxy-data setting.
