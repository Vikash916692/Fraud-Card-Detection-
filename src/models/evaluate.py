"""
Comprehensive model evaluation module for Credit Card Fraud Detection.

Computes production risk and fraud analytics metrics:
1. Precision (fraud class)
2. Recall (fraud class)
3. F1-Score (fraud class)
4. AUC-ROC
5. AUC-PR (Average Precision — primary metric under severe class imbalance)
6. Kolmogorov-Smirnov (KS) Statistic (standard risk-ranking metric)
7. Raw Confusion Matrix (TP, FP, TN, FN)
8. Visual artifacts (PR curve, ROC curve, Confusion Matrix) saved to models/eval_plots/
"""

from typing import Dict, Any, Tuple
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import EVAL_PLOTS_DIR

logger = logging.getLogger(__name__)


def calculate_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov (KS) statistic and its p-value.

    The KS statistic measures the maximum separation between the cumulative distribution
    function (CDF) of predicted probabilities for legitimate vs fraudulent transactions.

    Args:
        y_true: Ground truth binary labels (0 = Legit, 1 = Fraud).
        y_proba: Predicted probability of positive class (Fraud).

    Returns:
        Tuple[float, float]: (ks_statistic, p_value)
    """
    scores_legit = y_proba[y_true == 0]
    scores_fraud = y_proba[y_true == 1]
    
    if len(scores_legit) == 0 or len(scores_fraud) == 0:
        return 0.0, 1.0

    res = ks_2samp(scores_legit, scores_fraud)
    return float(res.statistic), float(res.pvalue)


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud evaluation metrics.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted fraud probabilities (floats between 0 and 1).
        threshold: Classification decision threshold (default: 0.5).

    Returns:
        Dict[str, Any]: Metrics dictionary.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    # Classification metrics for fraud class (Class = 1)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Ranking metrics
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    ks_stat, ks_pval = calculate_ks_statistic(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_test_samples": int(len(y_true)),
        "actual_fraud_samples": int(np.sum(y_true)),
    }

    return metrics


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "XGBoost",
    output_dir: Path = EVAL_PLOTS_DIR,
) -> Dict[str, Path]:
    """
    Generate and save high-resolution evaluation curve plots:
    1. Precision-Recall Curve (with AUC-PR & baseline fraud rate).
    2. Receiver Operating Characteristic (ROC) Curve.
    3. Confusion Matrix Visualization.

    Returns:
        Dict[str, Path]: Mapping of plot names to saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    saved_plots = {}

    # 1. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    base_rate = np.sum(y_true) / len(y_true)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall, precision, color="#1f77b4", lw=2.5, label=f"{model_name} (AUC-PR = {auc_pr:.4f})")
    plt.axhline(y=base_rate, color="red", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({base_rate:.4f})")
    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = output_dir / "pr_curve.png"
    plt.savefig(pr_path)
    plt.close()
    saved_plots["pr_curve"] = pr_path

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"{model_name} (AUC-ROC = {auc_roc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (0.5000)")
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="bold")
    plt.title(f"Receiver Operating Characteristic (ROC) — {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    saved_plots["roc_curve"] = roc_path

    # 3. Confusion Matrix
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold=0.5) — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Legitimate (0)", "Fraud (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Legitimate (0)", "Fraud (1)"], fontsize=10)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    saved_plots["confusion_matrix"] = cm_path

    logger.info(f"Evaluation plots saved successfully in {output_dir}")
    return saved_plots


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation on a test set, log results, and optionally export plots.
    """
    logger.info(f"Evaluating {model_name} on test set ({len(X_test):,} samples)...")
    
    # Get probability of fraud
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Sigmoid transform for margin scores
        y_proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_proba = model.predict(X_test)

    metrics = compute_all_metrics(y_test.values, y_proba)
    metrics["model_name"] = model_name

    logger.info(
        f"[{model_name}] Metrics: "
        f"AUC-PR: {metrics['auc_pr']:.4f} | "
        f"AUC-ROC: {metrics['auc_roc']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1_score']:.4f} | "
        f"KS-Stat: {metrics['ks_statistic']:.4f}"
    )

    if generate_plots:
        plot_evaluation_curves(y_test.values, y_proba, model_name=model_name)

    return metrics
