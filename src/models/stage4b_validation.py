from __future__ import annotations

from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, TABLES_DIR
from src.utils.io import write_text


RANDOM_STATE = 42
RANDOM_FOREST_NAME = "Random Forest"
XGBOOST_NAME = "XGBoost"
ISOLATION_FOREST_NAME = "Isolation Forest"
AUTOENCODER_NAME = "Autoencoder"

# Calibrated thresholds from Stage 4
CALIBRATED_THRESHOLDS = {
    RANDOM_FOREST_NAME: 0.527,
    XGBOOST_NAME: 0.415,
    ISOLATION_FOREST_NAME: 0.576,
    AUTOENCODER_NAME: 0.878,
}


def load_frame(file_name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / file_name, compression="gzip")


def load_stage_3_full_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load full dataset from Stage 3 (combined train+valid) for stratified CV."""
    x_train_supervised = load_frame("03_X_train_supervised.csv.gz")
    x_valid_supervised = load_frame("03_X_valid_supervised.csv.gz")
    y_train_supervised = load_frame("03_y_train_supervised.csv.gz")["Class"]
    y_valid_supervised = load_frame("03_y_valid_supervised.csv.gz")["Class"]

    # Combine for stratified CV
    X_combined = pd.concat([x_train_supervised, x_valid_supervised], ignore_index=True)
    y_combined = pd.concat([y_train_supervised, y_valid_supervised], ignore_index=True)

    # Also load unsupervised (normal-only) data for Isolation Forest and Autoencoder
    X_train_unsupervised_normal = load_frame("03_X_train_unsupervised_normal.csv.gz")

    class_weights_df = pd.read_csv(TABLES_DIR / "03_class_weights.csv")

    return X_combined, y_combined, X_train_unsupervised_normal, class_weights_df


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute comprehensive metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_rf_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_weights: dict[int, float],
    random_state: int,
) -> tuple[dict[str, float], float]:
    """Train Random Forest on fold."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight={int(k): v for k, v in class_weights.items()},
        min_samples_leaf=1,
        max_features="sqrt",
    )
    start = perf_counter()
    model.fit(X_train, y_train)
    train_time = perf_counter() - start
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= CALIBRATED_THRESHOLDS[RANDOM_FOREST_NAME]).astype(int)
    metrics = compute_metrics(y_test, predictions, probabilities)
    return metrics, train_time


def train_xgb_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> tuple[dict[str, float], float]:
    """Train XGBoost on fold."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)
    
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    start = perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = perf_counter() - start
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= CALIBRATED_THRESHOLDS[XGBOOST_NAME]).astype(int)
    metrics = compute_metrics(y_test, predictions, probabilities)
    return metrics, train_time


def train_if_fold(
    X_train_normal: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> tuple[dict[str, float], float]:
    """Train Isolation Forest on fold."""
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    start = perf_counter()
    model.fit(X_train_normal)
    train_time = perf_counter() - start
    scores = -model.score_samples(X_test)
    predictions = (scores >= CALIBRATED_THRESHOLDS[ISOLATION_FOREST_NAME]).astype(int)
    metrics = compute_metrics(y_test, predictions, scores)
    return metrics, train_time


def train_ae_fold(
    X_train_normal: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> tuple[dict[str, float], float]:
    """Train Autoencoder on fold."""
    model = MLPRegressor(
        hidden_layer_sizes=(24, 12, 24),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=80,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
    )
    start = perf_counter()
    model.fit(X_train_normal, X_train_normal)
    train_time = perf_counter() - start
    
    test_recon = model.predict(X_test)
    errors = np.mean((X_test.to_numpy() - test_recon) ** 2, axis=1)
    predictions = (errors >= CALIBRATED_THRESHOLDS[AUTOENCODER_NAME]).astype(int)
    metrics = compute_metrics(y_test, predictions, errors)
    return metrics, train_time


def run_stage_4b_validation() -> None:
    """Execute repeated stratified K-fold CV with multiple seeds."""
    print("[Stage 4b] Loading combined data for stratified CV...")
    X_combined, y_combined, X_train_normal, class_weights_df = load_stage_3_full_data()
    
    class_weight_map = dict(zip(class_weights_df["class_label"], class_weights_df["class_weight"], strict=False))
    
    # Stratified CV: 5 folds × 3 seeds = 15 total iterations
    random_seeds = [42, 123, 456]
    n_splits = 5
    
    all_results = []

    for seed in random_seeds:
        print(f"[Stage 4b] Processing seed {seed}...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
            print(f"  [Fold {fold_idx + 1}/{n_splits}]", end="", flush=True)
            
            X_train, X_test = X_combined.iloc[train_idx], X_combined.iloc[test_idx]
            y_train, y_test = y_combined.iloc[train_idx], y_combined.iloc[test_idx]
            
            # For unsupervised: use normal samples from train set
            X_train_normal_fold = X_train[y_train == 0]
            
            # Train Random Forest
            rf_metrics, rf_time = train_rf_fold(X_train, y_train, X_test, y_test, class_weight_map, seed)
            all_results.append({
                "seed": seed,
                "fold": fold_idx + 1,
                "model": RANDOM_FOREST_NAME,
                **rf_metrics,
                "training_seconds": rf_time,
            })
            
            # Train XGBoost
            xgb_metrics, xgb_time = train_xgb_fold(X_train, y_train, X_test, y_test, seed)
            all_results.append({
                "seed": seed,
                "fold": fold_idx + 1,
                "model": XGBOOST_NAME,
                **xgb_metrics,
                "training_seconds": xgb_time,
            })
            
            # Train Isolation Forest
            if_metrics, if_time = train_if_fold(X_train_normal_fold, X_test, y_test, seed)
            all_results.append({
                "seed": seed,
                "fold": fold_idx + 1,
                "model": ISOLATION_FOREST_NAME,
                **if_metrics,
                "training_seconds": if_time,
            })
            
            # Train Autoencoder
            ae_metrics, ae_time = train_ae_fold(X_train_normal_fold, X_test, y_test, seed)
            all_results.append({
                "seed": seed,
                "fold": fold_idx + 1,
                "model": AUTOENCODER_NAME,
                **ae_metrics,
                "training_seconds": ae_time,
            })
            
            print(" ✓")

    # Save all fold results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / "04b_cv_fold_results.csv", index=False)
    print(f"[Stage 4b] Saved fold results: {TABLES_DIR / '04b_cv_fold_results.csv'}")

    # Compute statistics per model
    stats_records = []
    for model_name in [RANDOM_FOREST_NAME, XGBOOST_NAME, ISOLATION_FOREST_NAME, AUTOENCODER_NAME]:
        model_results = results_df[results_df["model"] == model_name]
        
        for metric in ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
            mean_val = model_results[metric].mean()
            std_val = model_results[metric].std()
            min_val = model_results[metric].min()
            max_val = model_results[metric].max()
            
            stats_records.append({
                "model": model_name,
                "metric": metric,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            })

    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(TABLES_DIR / "04b_cv_statistics.csv", index=False)
    print(f"[Stage 4b] Saved statistics: {TABLES_DIR / '04b_cv_statistics.csv'}")

    # Save predictions for statistical testing
    results_df.to_csv(TABLES_DIR / "04b_cv_predictions.csv", index=False)

    # Write summary report
    summary_lines = [
        "Repeated Stratified K-Fold Validation Results",
        "=" * 60,
        f"Configuration: {n_splits} folds × {len(random_seeds)} seeds = {n_splits * len(random_seeds)} total runs per model",
        f"Random seeds: {random_seeds}",
        "",
        "Mean Performance Across All Folds:",
        "-" * 60,
    ]

    for model_name in [RANDOM_FOREST_NAME, XGBOOST_NAME, ISOLATION_FOREST_NAME, AUTOENCODER_NAME]:
        model_results = results_df[results_df["model"] == model_name]
        summary_lines.extend([
            f"\n{model_name}:",
            f"  Precision: {model_results['precision'].mean():.6f} ± {model_results['precision'].std():.6f}",
            f"  Recall:    {model_results['recall'].mean():.6f} ± {model_results['recall'].std():.6f}",
            f"  F1-Score:  {model_results['f1_score'].mean():.6f} ± {model_results['f1_score'].std():.6f}",
            f"  ROC-AUC:   {model_results['roc_auc'].mean():.6f} ± {model_results['roc_auc'].std():.6f}",
            f"  PR-AUC:    {model_results['pr_auc'].mean():.6f} ± {model_results['pr_auc'].std():.6f}",
        ])

    summary_lines.extend([
        "",
        "Notes:",
        "- Each fold is stratified to maintain fraud/normal ratio.",
        "- Multiple seeds provide variance estimates (std) across different data splits.",
        "- This enables statistical significance testing for model comparisons.",
        "- Predictions stored in 04b_cv_predictions.csv for McNemar's test.",
    ])

    write_text(TABLES_DIR / "04b_cv_summary.txt", "\n".join(summary_lines))
    print(f"[Stage 4b] Saved summary: {TABLES_DIR / '04b_cv_summary.txt'}")
    print("[Stage 4b] Validation complete.\n")


if __name__ == "__main__":
    run_stage_4b_validation()
