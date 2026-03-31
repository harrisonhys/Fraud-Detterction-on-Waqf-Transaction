from __future__ import annotations

from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR, REPORTS_DIR, TABLES_DIR
from src.utils.io import write_text


sns.set_theme(style="whitegrid")


def load_stage_4_metrics() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "04_model_metrics.csv")


def build_comparison_table(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    comparison_frame = metrics_frame.copy()
    comparison_frame["false_positive_rate"] = comparison_frame["fp"] / (comparison_frame["fp"] + comparison_frame["tn"])
    comparison_frame["false_negative_rate"] = comparison_frame["fn"] / (comparison_frame["fn"] + comparison_frame["tp"])
    comparison_frame["precision_rank"] = comparison_frame["precision"].rank(ascending=False, method="min").astype(int)
    comparison_frame["recall_rank"] = comparison_frame["recall"].rank(ascending=False, method="min").astype(int)
    comparison_frame["f1_rank"] = comparison_frame["f1_score"].rank(ascending=False, method="min").astype(int)
    comparison_frame["roc_auc_rank"] = comparison_frame["roc_auc"].rank(ascending=False, method="min").astype(int)
    comparison_frame["pr_auc_rank"] = comparison_frame["pr_auc"].rank(ascending=False, method="min").astype(int)
    comparison_frame["overall_score"] = (
        0.30 * comparison_frame["f1_score"]
        + 0.25 * comparison_frame["pr_auc"]
        + 0.20 * comparison_frame["recall"]
        + 0.15 * comparison_frame["precision"]
        + 0.10 * comparison_frame["roc_auc"]
    )
    comparison_frame["overall_rank"] = comparison_frame["overall_score"].rank(ascending=False, method="min").astype(int)
    return comparison_frame.sort_values("overall_rank").reset_index(drop=True)


def save_stage_5_plot(comparison_frame: pd.DataFrame) -> None:
    plot_frame = comparison_frame.melt(
        id_vars=["model_name"],
        value_vars=["precision", "recall", "f1_score", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="score",
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_frame, x="metric", y="score", hue="model_name", ax=axis)
    axis.set_title("Comparative Fraud Detection Performance")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "05_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_comparative_analysis(comparison_frame: pd.DataFrame) -> None:
    best_model = comparison_frame.iloc[0]
    second_model = comparison_frame.iloc[1]
    weakest_model = comparison_frame.iloc[-1]
    unsupervised_frame = comparison_frame[comparison_frame["model_name"].isin(["Isolation Forest", "Autoencoder"])]
    best_unsupervised = unsupervised_frame.sort_values("f1_score", ascending=False).iloc[0]

    report = dedent(
        f"""
        Stage: Evaluation Comparative Analysis

        Comparative summary:
        1. {best_model['model_name']} ranked first overall based on the weighted evaluation score, supported by the highest F1-score and the highest PR-AUC.
        2. {second_model['model_name']} ranked second and achieved the strongest raw precision, indicating especially tight control of false positives.
        3. {weakest_model['model_name']} ranked last because, although its ROC-AUC remained reasonably high, its precision and PR-AUC were not competitive in the extreme-imbalance setting.

        Metric-level interpretation:
        1. XGBoost achieved the strongest balance of precision ({comparison_frame.loc[comparison_frame['model_name'] == 'XGBoost', 'precision'].iloc[0]:.6f}) and recall ({comparison_frame.loc[comparison_frame['model_name'] == 'XGBoost', 'recall'].iloc[0]:.6f}), resulting in the best F1-score ({comparison_frame.loc[comparison_frame['model_name'] == 'XGBoost', 'f1_score'].iloc[0]:.6f}).
        2. Random Forest delivered the lowest false positive burden with only {int(comparison_frame.loc[comparison_frame['model_name'] == 'Random Forest', 'fp'].iloc[0])} false positives, but it missed more frauds than XGBoost.
        3. Among unsupervised models, {best_unsupervised['model_name']} provided the strongest fraud-capture trade-off, but unsupervised precision remained substantially lower than supervised alternatives.

        False positive versus false negative trade-off:
        1. Random Forest is conservative: very few legitimate transactions are flagged, but more fraud cases remain undetected.
        2. XGBoost is more balanced: it accepts a small increase in false positives to capture more fraud cases.
        3. Isolation Forest and Autoencoder are suitable mainly for exploratory anomaly surfacing and drift monitoring rather than primary high-precision intervention.

        Evaluation conclusion:
        In this proxy-dataset experiment, supervised learning clearly outperforms unsupervised anomaly detection for transaction-level fraud classification. XGBoost is the strongest candidate for the primary model in the research narrative, while Random Forest is a credible precision-oriented benchmark and Autoencoder with Isolation Forest remain secondary anomaly-monitoring tools.
        """
    )
    write_text(REPORTS_DIR / "05_evaluation_comparative_analysis.txt", report)


def write_waqf_interpretation(comparison_frame: pd.DataFrame) -> None:
    xgboost_row = comparison_frame.loc[comparison_frame["model_name"] == "XGBoost"].iloc[0]
    random_forest_row = comparison_frame.loc[comparison_frame["model_name"] == "Random Forest"].iloc[0]
    unsupervised_frame = comparison_frame[comparison_frame["model_name"].isin(["Isolation Forest", "Autoencoder"])]
    best_unsupervised = unsupervised_frame.sort_values("f1_score", ascending=False).iloc[0]

    report = dedent(
        f"""
        Stage: Interpretation for Waqf Context

        Trust implications:
        1. A digital waqf platform needs strong fraud capture without over-penalizing legitimate donors or administrators.
        2. XGBoost offers the most credible trust-preserving balance in this study because it captures {int(xgboost_row['tp'])} fraudulent cases while keeping false positives limited to {int(xgboost_row['fp'])}.
        3. Random Forest is valuable where governance policy prioritizes minimizing unnecessary donor friction, since it produced only {int(random_forest_row['fp'])} false positives.

        Transparency implications:
        1. Tree-based models provide an auditable basis for monitoring suspicious transaction behavior through feature importance ranking.
        2. Both Random Forest and XGBoost identified similar leading signals, especially V14, V10, V12, and V4, which strengthens confidence that the supervised models are learning stable discriminatory patterns in the proxy dataset.
        3. In an actual waqf deployment, analogous interpretable features could include abnormal donation amount, transaction timing irregularity, repeated small transfers, donor account volatility, or device inconsistency.

        Governance implications:
        1. If the operational objective is early fraud interception with acceptable review effort, XGBoost is the best candidate for first-line automated scoring.
        2. If the objective is stricter protection against false accusations, Random Forest can serve as a complementary or secondary confirmation model.
        3. {best_unsupervised['model_name']} is better positioned as an anomaly watchlist mechanism for emerging patterns that are not yet represented in labeled historical data.

        Recommended research narrative:
        1. Supervised learning is more effective when labeled fraud-like proxy data exists.
        2. Unsupervised anomaly detection via Isolation Forest and Autoencoder remains strategically useful for identifying novel or evolving suspicious behaviors in digital waqf ecosystems.
        3. A layered monitoring design for digital waqf could combine a high-performing supervised model for primary screening and an anomaly model for secondary exploratory surveillance.

        Waqf-context caution:
        The current results should not be interpreted as direct operational performance for waqf transactions. They instead demonstrate how machine learning methods can support trust, transparency, and governance design in the absence of a public waqf fraud dataset.
        """
    )
    write_text(REPORTS_DIR / "05_interpretation_for_waqf_context.txt", report)


def write_limitations_report(comparison_frame: pd.DataFrame) -> None:
    best_model = comparison_frame.iloc[0]
    report = dedent(
        f"""
        Stage: Limitations and Validity

        Proxy-data limitations:
        1. The experiment uses a credit card fraud dataset rather than actual waqf transaction records.
        2. Most predictor variables are anonymized, limiting semantic mapping from model behavior to specific waqf operational processes.
        3. The fraud generation process in card payments may differ materially from misconduct patterns in waqf donation, management, or disbursement workflows.

        Internal validity considerations:
        1. The pipeline is reproducible, uses a fixed random seed, and relies on a stratified validation split.
        2. Duplicate rows were removed before modeling to reduce evaluation bias.
        3. The strongest model in this setting was {best_model['model_name']}, but its reported advantage is conditional on the current split, threshold, and feature representation.

        External validity considerations:
        1. Real digital waqf systems may include metadata unavailable in this dataset, such as donor identity, beneficiary category, institution role, channel, geography, and device traces.
        2. Base fraud rates in waqf ecosystems may be lower or structurally different from those in card transactions.
        3. Model transfer to real waqf environments would require domain adaptation, threshold recalibration, governance review, and human-in-the-loop validation.

        Practical validity guidance:
        1. The present study supports method comparison, evaluation design, and governance framing.
        2. It does not justify direct production deployment in waqf platforms without real-domain validation.
        3. Future work should collect or simulate waqf-specific transaction data, include richer contextual variables, and evaluate concept drift over time.
        """
    )
    write_text(REPORTS_DIR / "05_limitations_and_validity.txt", report)


def write_ai_ready_summary(comparison_frame: pd.DataFrame) -> None:
    lines = [
        "AI-Ready Evaluation Summary",
        "",
        "Model comparison ranking:",
    ]
    for record in comparison_frame.to_dict(orient="records"):
        lines.append(
            f"- rank={record['overall_rank']} model={record['model_name']} overall_score={record['overall_score']:.6f} precision={record['precision']:.6f} recall={record['recall']:.6f} f1={record['f1_score']:.6f} roc_auc={record['roc_auc']:.6f} pr_auc={record['pr_auc']:.6f} fp={int(record['fp'])} fn={int(record['fn'])}"
        )
    lines.extend(
        [
            "",
            "Key conclusions:",
            "- XGBoost is the strongest overall candidate for primary fraud detection in the proxy setting.",
            "- Random Forest is attractive when minimizing false positives is a primary operational objective.",
            "- Isolation Forest and Autoencoder are useful mainly as anomaly-monitoring complements rather than primary classifiers.",
            "- Findings support ML-based trust and governance enhancement for digital waqf, but only as proxy evidence.",
        ]
    )
    write_text(REPORTS_DIR / "05_ai_ready_evaluation_summary.txt", "\n".join(lines))


def run_stage_5_evaluation() -> None:
    metrics_frame = load_stage_4_metrics()
    comparison_frame = build_comparison_table(metrics_frame)
    comparison_frame.to_csv(TABLES_DIR / "05_final_model_comparison.csv", index=False)
    save_stage_5_plot(comparison_frame)
    write_comparative_analysis(comparison_frame)
    write_waqf_interpretation(comparison_frame)
    write_limitations_report(comparison_frame)
    write_ai_ready_summary(comparison_frame)