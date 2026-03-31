from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, REPORTS_DIR, TABLES_DIR
from src.utils.io import write_text


sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
RANDOM_FOREST_NAME = "Random Forest"
XGBOOST_NAME = "XGBoost"
ISOLATION_FOREST_NAME = "Isolation Forest"
AUTOENCODER_NAME = "Autoencoder"


@dataclass
class ModelArtifacts:
    name: str
    model: object
    probabilities_or_scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    threshold_logic: str
    threshold_objective: str
    training_seconds: float
    parameters: dict[str, object]
    feature_importance: pd.DataFrame | None = None


def load_frame(file_name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / file_name, compression="gzip")


def load_stage_3_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    x_train = load_frame("03_X_train_supervised.csv.gz")
    x_valid = load_frame("03_X_valid_supervised.csv.gz")
    y_train = load_frame("03_y_train_supervised.csv.gz")["Class"]
    y_valid = load_frame("03_y_valid_supervised.csv.gz")["Class"]
    x_train_normal = load_frame("03_X_train_unsupervised_normal.csv.gz")
    class_weights = pd.read_csv(TABLES_DIR / "03_class_weights.csv")
    return x_train, x_valid, y_train, y_valid, x_train_normal, class_weights


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
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


def evaluate_threshold_grid(y_true: pd.Series, y_score: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
            }
        )
    return pd.DataFrame(rows)


def calibrate_supervised_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    objective: str,
    min_constraint: float,
    model_name: str,
) -> tuple[float, str, pd.DataFrame]:
    quantile_grid = np.linspace(0.01, 0.99, 199)
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 1001),
                np.quantile(y_score, quantile_grid),
            ]
        )
    )
    calibration_frame = evaluate_threshold_grid(y_true, y_score, candidate_thresholds)
    calibration_frame["model_name"] = model_name

    if objective == "precision_priority":
        beta = 0.5
        constrained = calibration_frame[calibration_frame["recall"] >= min_constraint].copy()
        if not constrained.empty:
            constrained["objective_score"] = (
                (1 + beta**2) * constrained["precision"] * constrained["recall"]
            ) / ((beta**2 * constrained["precision"]) + constrained["recall"] + 1e-12)
            chosen = constrained.sort_values(["objective_score", "precision"], ascending=[False, False]).iloc[0]
            logic = (
                f"Precision-priority calibration: selected threshold maximizing F0.5 under recall >= {min_constraint:.2f}."
            )
        else:
            calibration_frame["objective_score"] = (
                (1 + beta**2) * calibration_frame["precision"] * calibration_frame["recall"]
            ) / ((beta**2 * calibration_frame["precision"]) + calibration_frame["recall"] + 1e-12)
            chosen = calibration_frame.sort_values(["objective_score", "precision"], ascending=[False, False]).iloc[0]
            logic = (
                f"Precision-priority calibration fallback: no threshold met recall >= {min_constraint:.2f}, "
                "so threshold maximizing F0.5 was selected."
            )
    elif objective == "recall_priority":
        beta = 2.0
        constrained = calibration_frame[calibration_frame["precision"] >= min_constraint].copy()
        if constrained.empty:
            calibration_frame["objective_score"] = (
                (1 + beta**2) * calibration_frame["precision"] * calibration_frame["recall"]
            ) / ((beta**2 * calibration_frame["precision"]) + calibration_frame["recall"] + 1e-12)
            chosen = calibration_frame.sort_values(["objective_score", "recall"], ascending=[False, False]).iloc[0]
            logic = (
                f"Recall-priority calibration fallback: no threshold met precision >= {min_constraint:.2f}, "
                "so threshold maximizing F2 was selected."
            )
        else:
            constrained["objective_score"] = (
                (1 + beta**2) * constrained["precision"] * constrained["recall"]
            ) / ((beta**2 * constrained["precision"]) + constrained["recall"] + 1e-12)
            chosen = constrained.sort_values(["objective_score", "recall"], ascending=[False, False]).iloc[0]
            logic = (
                f"Recall-priority calibration: selected threshold maximizing F2 under precision >= {min_constraint:.2f}."
            )
    else:
        chosen = calibration_frame.sort_values("f1_score", ascending=False).iloc[0]
        logic = "F1-priority calibration: selected threshold maximizing F1-score."

    return float(chosen["threshold"]), logic, calibration_frame


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, path_name: str, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axis, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / path_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_plot(y_true: pd.Series, y_score: np.ndarray, path_name: str, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", color="#1D3557")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / path_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_pr_plot(y_true: pd.Series, y_score: np.ndarray, path_name: str, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", color="#D1495B")
    axis.set_title(title)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / path_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_feature_importance_plot(feature_importance: pd.DataFrame, path_name: str, title: str) -> None:
    top_features = feature_importance.sort_values("importance", ascending=False).head(15)
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top_features, x="importance", y="feature", color="#2A9D8F", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / path_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_isolation_score_plot(
    train_scores: np.ndarray,
    valid_scores: np.ndarray,
    y_valid: pd.Series,
    threshold: float,
    path_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(train_scores, bins=60, stat="density", color="#4C956C", label="Train Normal", ax=axis, alpha=0.5)
    sns.histplot(valid_scores[y_valid.to_numpy() == 0], bins=60, stat="density", color="#457B9D", label="Valid Non-Fraud", ax=axis, alpha=0.5)
    sns.histplot(valid_scores[y_valid.to_numpy() == 1], bins=40, stat="density", color="#D1495B", label="Valid Fraud", ax=axis, alpha=0.5)
    axis.axvline(threshold, color="#111111", linestyle="--", label=f"Threshold = {threshold:.4f}")
    axis.set_title("Isolation Forest Anomaly Score Distribution")
    axis.set_xlabel("Anomaly Score")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / path_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_score_plot(
    train_scores: np.ndarray,
    valid_scores: np.ndarray,
    y_valid: pd.Series,
    threshold: float,
    path_name: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(train_scores, bins=60, stat="density", color="#4C956C", label="Train Normal", ax=axis, alpha=0.5)
    sns.histplot(valid_scores[y_valid.to_numpy() == 0], bins=60, stat="density", color="#457B9D", label="Valid Non-Fraud", ax=axis, alpha=0.5)
    sns.histplot(valid_scores[y_valid.to_numpy() == 1], bins=40, stat="density", color="#D1495B", label="Valid Fraud", ax=axis, alpha=0.5)
    axis.axvline(threshold, color="#111111", linestyle="--", label=f"Threshold = {threshold:.4f}")
    axis.set_title(title)
    axis.set_xlabel("Reconstruction Error")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / path_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metric_comparison_plot(metrics_frame: pd.DataFrame) -> None:
    plot_frame = metrics_frame.melt(
        id_vars=["model_name"],
        value_vars=["precision", "recall", "f1_score", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="value",
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_frame, x="metric", y="value", hue="model_name", ax=axis)
    axis.set_title("Stage 4 Model Metric Comparison")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "04_model_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_threshold_calibration_plot(calibration_frame: pd.DataFrame, model_name: str, selected_threshold: float) -> None:
    frame = calibration_frame[calibration_frame["model_name"] == model_name].copy()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["threshold"], frame["precision"], label="Precision", color="#1D3557")
    axis.plot(frame["threshold"], frame["recall"], label="Recall", color="#D1495B")
    axis.plot(frame["threshold"], frame["f1_score"], label="F1", color="#2A9D8F")
    axis.axvline(selected_threshold, linestyle="--", color="#111111", label=f"Selected = {selected_threshold:.4f}")
    axis.set_title(f"{model_name} Threshold Calibration")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.legend()
    figure.tight_layout()
    file_name = f"04_{model_name.lower().replace(' ', '_')}_threshold_calibration.png"
    figure.savefig(PLOTS_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_model_report(model_name: str, artifacts: ModelArtifacts, metrics: dict[str, float], extra_notes: str) -> None:
    feature_section = "No feature importance available for this model."
    if artifacts.feature_importance is not None:
        top_features = artifacts.feature_importance.sort_values("importance", ascending=False).head(10)
        feature_section = top_features.to_string(index=False)

    report = dedent(
        f"""
        Stage: Modeling
        Model: {model_name}

        Training summary:
        - Training time (seconds): {artifacts.training_seconds:.2f}
        - Threshold: {artifacts.threshold:.6f}
        - Threshold logic: {artifacts.threshold_logic}

        Parameters:
        {artifacts.parameters}

        Evaluation metrics:
        - Precision: {metrics['precision']:.6f}
        - Recall: {metrics['recall']:.6f}
        - F1-score: {metrics['f1_score']:.6f}
        - ROC-AUC: {metrics['roc_auc']:.6f}
        - PR-AUC: {metrics['pr_auc']:.6f}
        - TN: {metrics['tn']}
        - FP: {metrics['fp']}
        - FN: {metrics['fn']}
        - TP: {metrics['tp']}

        Feature importance snapshot:
        {feature_section}

        Notes:
        {extra_notes}
        """
    )
    report_path = REPORTS_DIR / f"04_modeling_{model_name.lower().replace(' ', '_')}.txt"
    write_text(report_path, report)


def train_random_forest(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    class_weights: pd.DataFrame,
) -> tuple[ModelArtifacts, dict[str, float]]:
    class_weight_map = dict(zip(class_weights["class_label"], class_weights["class_weight"], strict=False))
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight_map,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    start_time = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - start_time
    probabilities = model.predict_proba(x_valid)[:, 1]
    threshold, threshold_logic, calibration_frame = calibrate_supervised_threshold(
        y_true=y_valid,
        y_score=probabilities,
        objective="precision_priority",
        min_constraint=0.30,
        model_name=RANDOM_FOREST_NAME,
    )
    predictions = (probabilities >= threshold).astype(int)
    feature_importance = pd.DataFrame({"feature": x_train.columns, "importance": model.feature_importances_})
    artifacts = ModelArtifacts(
        name=RANDOM_FOREST_NAME,
        model=model,
        probabilities_or_scores=probabilities,
        predictions=predictions,
        threshold=threshold,
        threshold_logic=threshold_logic,
        threshold_objective="precision_priority",
        training_seconds=training_seconds,
        parameters=model.get_params(),
        feature_importance=feature_importance,
    )
    return artifacts, compute_metrics(y_valid, predictions, probabilities), calibration_frame


def train_xgboost(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> tuple[ModelArtifacts, dict[str, float]]:
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
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    start_time = perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    training_seconds = perf_counter() - start_time
    probabilities = model.predict_proba(x_valid)[:, 1]
    threshold, threshold_logic, calibration_frame = calibrate_supervised_threshold(
        y_true=y_valid,
        y_score=probabilities,
        objective="recall_priority",
        min_constraint=0.10,
        model_name=XGBOOST_NAME,
    )
    predictions = (probabilities >= threshold).astype(int)
    feature_importance = pd.DataFrame({"feature": x_train.columns, "importance": model.feature_importances_})
    artifacts = ModelArtifacts(
        name=XGBOOST_NAME,
        model=model,
        probabilities_or_scores=probabilities,
        predictions=predictions,
        threshold=threshold,
        threshold_logic=threshold_logic,
        threshold_objective="recall_priority",
        training_seconds=training_seconds,
        parameters=model.get_params(),
        feature_importance=feature_importance,
    )
    return artifacts, compute_metrics(y_valid, predictions, probabilities), calibration_frame


def train_isolation_forest(
    x_train_normal: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[ModelArtifacts, dict[str, float], np.ndarray]:
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    start_time = perf_counter()
    model.fit(x_train_normal)
    training_seconds = perf_counter() - start_time

    train_scores = -model.score_samples(x_train_normal)
    valid_scores = -model.score_samples(x_valid)
    threshold = float(np.quantile(train_scores, 0.99))
    predictions = (valid_scores >= threshold).astype(int)
    artifacts = ModelArtifacts(
        name=ISOLATION_FOREST_NAME,
        model=model,
        probabilities_or_scores=valid_scores,
        predictions=predictions,
        threshold=threshold,
        threshold_logic="Anomaly threshold set at the 99th percentile of anomaly scores from normal-only training data.",
        threshold_objective="anomaly_quantile_99",
        training_seconds=training_seconds,
        parameters=model.get_params(),
        feature_importance=None,
    )
    return artifacts, compute_metrics(y_valid, predictions, valid_scores), train_scores


def train_autoencoder(
    x_train_normal: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[ModelArtifacts, dict[str, float], np.ndarray]:
    model = MLPRegressor(
        hidden_layer_sizes=(24, 12, 24),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=80,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
    )
    start_time = perf_counter()
    model.fit(x_train_normal, x_train_normal)
    training_seconds = perf_counter() - start_time

    train_recon = model.predict(x_train_normal)
    valid_recon = model.predict(x_valid)
    train_errors = np.mean((x_train_normal.to_numpy() - train_recon) ** 2, axis=1)
    valid_errors = np.mean((x_valid.to_numpy() - valid_recon) ** 2, axis=1)

    threshold = float(np.quantile(train_errors, 0.99))
    predictions = (valid_errors >= threshold).astype(int)
    artifacts = ModelArtifacts(
        name=AUTOENCODER_NAME,
        model=model,
        probabilities_or_scores=valid_errors,
        predictions=predictions,
        threshold=threshold,
        threshold_logic="Reconstruction-error threshold set at the 99th percentile of errors from normal-only training data.",
        threshold_objective="reconstruction_quantile_99",
        training_seconds=training_seconds,
        parameters=model.get_params(),
        feature_importance=None,
    )
    return artifacts, compute_metrics(y_valid, predictions, valid_errors), train_errors


def write_ai_ready_summary(metrics_frame: pd.DataFrame) -> None:
    summary_lines = [
        "AI-Ready Modeling Summary",
        "",
        "Stage 4 models trained:",
    ]
    for record in metrics_frame.to_dict(orient="records"):
        summary_lines.append(
            f"- {record['model_name']}: precision={record['precision']:.6f}, recall={record['recall']:.6f}, f1={record['f1_score']:.6f}, roc_auc={record['roc_auc']:.6f}, pr_auc={record['pr_auc']:.6f}, threshold={record['threshold']:.6f}, train_seconds={record['training_seconds']:.2f}"
        )
    summary_lines.extend(
        [
            "",
            "Interpretive notes:",
            "- Random Forest and XGBoost use labeled supervised data from Stage 3.",
            "- Supervised thresholds are calibrated on validation probabilities using narrative-aligned objectives: Random Forest precision-priority and XGBoost recall-priority.",
            "- Isolation Forest is trained only on normal transactions and evaluated on labeled validation data.",
            "- Autoencoder is trained as an unsupervised reconstruction model on normal transactions and flags high reconstruction errors as anomalies.",
            "- Precision, recall, F1, ROC-AUC, and PR-AUC are prioritized because the fraud rate is extremely low.",
        ]
    )
    write_text(REPORTS_DIR / "04_ai_ready_modeling_summary.txt", "\n".join(summary_lines))


def run_stage_4_modeling() -> None:
    x_train, x_valid, y_train, y_valid, x_train_normal, class_weights = load_stage_3_inputs()

    random_forest_artifacts, random_forest_metrics, random_forest_calibration = train_random_forest(x_train, x_valid, y_train, y_valid, class_weights)
    xgboost_artifacts, xgboost_metrics, xgboost_calibration = train_xgboost(x_train, x_valid, y_train, y_valid)
    isolation_forest_artifacts, isolation_forest_metrics, isolation_train_scores = train_isolation_forest(x_train_normal, x_valid, y_valid)
    autoencoder_artifacts, autoencoder_metrics, autoencoder_train_errors = train_autoencoder(x_train_normal, x_valid, y_valid)

    calibration_frame = pd.concat([random_forest_calibration, xgboost_calibration], ignore_index=True)
    calibration_frame.to_csv(TABLES_DIR / "04_threshold_calibration.csv", index=False)

    joblib.dump(random_forest_artifacts.model, MODELS_DIR / "04_random_forest.joblib")
    joblib.dump(xgboost_artifacts.model, MODELS_DIR / "04_xgboost.joblib")
    joblib.dump(isolation_forest_artifacts.model, MODELS_DIR / "04_isolation_forest.joblib")
    joblib.dump(autoencoder_artifacts.model, MODELS_DIR / "04_autoencoder.joblib")

    save_confusion_matrix(y_valid, random_forest_artifacts.predictions, "04_random_forest_confusion_matrix.png", "Random Forest Confusion Matrix")
    save_roc_plot(y_valid, random_forest_artifacts.probabilities_or_scores, "04_random_forest_roc_curve.png", "Random Forest ROC Curve")
    save_pr_plot(y_valid, random_forest_artifacts.probabilities_or_scores, "04_random_forest_pr_curve.png", "Random Forest Precision-Recall Curve")
    save_feature_importance_plot(random_forest_artifacts.feature_importance, "04_random_forest_feature_importance.png", "Random Forest Feature Importance")
    save_threshold_calibration_plot(calibration_frame, RANDOM_FOREST_NAME, random_forest_artifacts.threshold)

    save_confusion_matrix(y_valid, xgboost_artifacts.predictions, "04_xgboost_confusion_matrix.png", "XGBoost Confusion Matrix")
    save_roc_plot(y_valid, xgboost_artifacts.probabilities_or_scores, "04_xgboost_roc_curve.png", "XGBoost ROC Curve")
    save_pr_plot(y_valid, xgboost_artifacts.probabilities_or_scores, "04_xgboost_pr_curve.png", "XGBoost Precision-Recall Curve")
    save_feature_importance_plot(xgboost_artifacts.feature_importance, "04_xgboost_feature_importance.png", "XGBoost Feature Importance")
    save_threshold_calibration_plot(calibration_frame, XGBOOST_NAME, xgboost_artifacts.threshold)

    save_confusion_matrix(y_valid, isolation_forest_artifacts.predictions, "04_isolation_forest_confusion_matrix.png", "Isolation Forest Confusion Matrix")
    save_roc_plot(y_valid, isolation_forest_artifacts.probabilities_or_scores, "04_isolation_forest_roc_curve.png", "Isolation Forest ROC Curve")
    save_pr_plot(y_valid, isolation_forest_artifacts.probabilities_or_scores, "04_isolation_forest_pr_curve.png", "Isolation Forest Precision-Recall Curve")
    save_isolation_score_plot(
        train_scores=isolation_train_scores,
        valid_scores=isolation_forest_artifacts.probabilities_or_scores,
        y_valid=y_valid,
        threshold=isolation_forest_artifacts.threshold,
        path_name="04_isolation_forest_anomaly_scores.png",
    )

    save_confusion_matrix(y_valid, autoencoder_artifacts.predictions, "04_autoencoder_confusion_matrix.png", "Autoencoder Confusion Matrix")
    save_roc_plot(y_valid, autoencoder_artifacts.probabilities_or_scores, "04_autoencoder_roc_curve.png", "Autoencoder ROC Curve")
    save_pr_plot(y_valid, autoencoder_artifacts.probabilities_or_scores, "04_autoencoder_pr_curve.png", "Autoencoder Precision-Recall Curve")
    save_reconstruction_score_plot(
        train_scores=autoencoder_train_errors,
        valid_scores=autoencoder_artifacts.probabilities_or_scores,
        y_valid=y_valid,
        threshold=autoencoder_artifacts.threshold,
        path_name="04_autoencoder_reconstruction_errors.png",
        title="Autoencoder Reconstruction Error Distribution",
    )

    write_model_report(
        model_name=RANDOM_FOREST_NAME,
        artifacts=random_forest_artifacts,
        metrics=random_forest_metrics,
        extra_notes="Class-balanced weighting was taken from Stage 3 outputs and applied directly to the classifier.",
    )
    write_model_report(
        model_name=XGBOOST_NAME,
        artifacts=xgboost_artifacts,
        metrics=xgboost_metrics,
        extra_notes="The model uses scale_pos_weight derived from the supervised training split to reflect the extreme class imbalance.",
    )
    write_model_report(
        model_name=ISOLATION_FOREST_NAME,
        artifacts=isolation_forest_artifacts,
        metrics=isolation_forest_metrics,
        extra_notes="The model is unsupervised during training and uses only normal transactions to learn the reference distribution.",
    )
    write_model_report(
        model_name=AUTOENCODER_NAME,
        artifacts=autoencoder_artifacts,
        metrics=autoencoder_metrics,
        extra_notes="The model is an unsupervised reconstruction autoencoder trained on normal transactions and uses reconstruction error for anomaly scoring.",
    )

    metrics_frame = pd.DataFrame(
        [
            {
                "model_name": random_forest_artifacts.name,
                **random_forest_metrics,
                "threshold": random_forest_artifacts.threshold,
                "threshold_objective": random_forest_artifacts.threshold_objective,
                "training_seconds": random_forest_artifacts.training_seconds,
            },
            {
                "model_name": xgboost_artifacts.name,
                **xgboost_metrics,
                "threshold": xgboost_artifacts.threshold,
                "threshold_objective": xgboost_artifacts.threshold_objective,
                "training_seconds": xgboost_artifacts.training_seconds,
            },
            {
                "model_name": isolation_forest_artifacts.name,
                **isolation_forest_metrics,
                "threshold": isolation_forest_artifacts.threshold,
                "threshold_objective": isolation_forest_artifacts.threshold_objective,
                "training_seconds": isolation_forest_artifacts.training_seconds,
            },
            {
                "model_name": autoencoder_artifacts.name,
                **autoencoder_metrics,
                "threshold": autoencoder_artifacts.threshold,
                "threshold_objective": autoencoder_artifacts.threshold_objective,
                "training_seconds": autoencoder_artifacts.training_seconds,
            },
        ]
    )
    metrics_frame.to_csv(TABLES_DIR / "04_model_metrics.csv", index=False)
    save_metric_comparison_plot(metrics_frame)
    write_ai_ready_summary(metrics_frame)