"""
SHAP (SHapley Additive exPlanations) explainability module for Credit Card Fraud Detection.

Generates:
1. Global feature importance beeswarm summary plot: models/eval_plots/shap_summary.png
2. Local prediction explanation (True Positive fraud case): models/eval_plots/shap_force_tp.png
3. Local prediction explanation (False Positive / False Negative case): models/eval_plots/shap_force_fp.png
4. Textual feature interpretation for risk and compliance review.
"""

from typing import Dict, Any, Tuple
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_DIR,
    EVAL_PLOTS_DIR,
    RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("shap_analysis")


def load_model_and_test_data() -> Tuple[Any, pd.DataFrame, pd.Series]:
    """Load the trained model and processed test dataset."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_ARTIFACT_PATH}")

    model = joblib.load(MODEL_ARTIFACT_PATH)

    # Load test split
    if (PROCESSED_DATA_DIR / "X_test.parquet").exists():
        X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
        y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet").iloc[:, 0]
    elif (PROCESSED_DATA_DIR / "X_test.csv").exists():
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0]
    else:
        raise FileNotFoundError("Processed test data not found in data/processed/")

    return model, X_test, y_test


def run_shap_analysis(
    sample_size: int = 2000,
    output_dir: Path = EVAL_PLOTS_DIR,
) -> Dict[str, Any]:
    """
    Execute TreeSHAP analysis, generate global and local explanation plots, and extract key features.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X_test, y_test = load_model_and_test_data()
    logger.info(f"Computing SHAP values for model: {type(model).__name__}...")

    # Sample a representative subset for efficient SHAP computation
    # Ensure both fraud and legit samples are included in sample
    fraud_indices = np.where(y_test.values == 1)[0]
    legit_indices = np.where(y_test.values == 0)[0]
    
    np.random.seed(RANDOM_STATE)
    sampled_legit = np.random.choice(
        legit_indices,
        size=min(sample_size - len(fraud_indices), len(legit_indices)),
        replace=False,
    )
    combined_indices = np.concatenate([fraud_indices, sampled_legit])
    np.random.shuffle(combined_indices)

    X_sample = X_test.iloc[combined_indices].reset_index(drop=True)
    y_sample = y_test.iloc[combined_indices].reset_index(drop=True)

    # Create TreeExplainer or generic Explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    except Exception as e:
        logger.warning(f"TreeExplainer failed ({e}), falling back to generic Explainer...")
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)

    # 1. Global Summary Beeswarm Plot
    logger.info("Generating SHAP global summary plot...")
    plt.figure(figsize=(10, 7), dpi=300)
    # If binary classification output has shape (N, D, 2), take slice for class 1 (Fraud)
    vals_for_plot = shap_values[..., 1] if len(shap_values.shape) == 3 else shap_values
    shap.summary_plot(vals_for_plot, X_sample, show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (Beeswarm)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    summary_plot_path = output_dir / "shap_summary.png"
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to {summary_plot_path}")

    # 2. Identify Cases for Local Force / Waterfall Plots
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    preds = (probs >= 0.5).astype(int)

    tp_indices = np.where((y_test.values == 1) & (preds == 1))[0]
    fp_indices = np.where((y_test.values == 0) & (preds == 1))[0]
    fn_indices = np.where((y_test.values == 1) & (preds == 0))[0]

    # Local Plot 1: True Positive (Confirmed Fraud caught by model)
    if len(tp_indices) > 0:
        tp_idx = tp_indices[0]
        tp_row = X_test.iloc[[tp_idx]]
        tp_shap = explainer(tp_row)
        tp_vals = tp_shap[0, :, 1] if len(tp_shap.shape) == 3 else tp_shap[0]

        plt.figure(figsize=(9, 6), dpi=300)
        shap.plots.waterfall(tp_vals, show=False, max_display=10)
        plt.title(f"SHAP Local Explanation: True Positive Fraud (Score: {probs[tp_idx]:.4f})", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        tp_plot_path = output_dir / "shap_force_tp.png"
        plt.savefig(tp_plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved True Positive SHAP waterfall plot to {tp_plot_path}")

    # Local Plot 2: False Positive or False Negative case
    case_idx = fp_indices[0] if len(fp_indices) > 0 else (fn_indices[0] if len(fn_indices) > 0 else tp_indices[-1])
    case_label = "False Positive" if len(fp_indices) > 0 else ("False Negative" if len(fn_indices) > 0 else "Boundary Case")
    case_row = X_test.iloc[[case_idx]]
    case_shap = explainer(case_row)
    case_vals = case_shap[0, :, 1] if len(case_shap.shape) == 3 else case_shap[0]

    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(case_vals, show=False, max_display=10)
    plt.title(f"SHAP Local Explanation: {case_label} (Score: {probs[case_idx]:.4f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fp_plot_path = output_dir / "shap_force_fp.png"
    plt.savefig(fp_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {case_label} SHAP waterfall plot to {fp_plot_path}")

    # Compute top 5 mean absolute SHAP features
    raw_shap_matrix = vals_for_plot.values if hasattr(vals_for_plot, "values") else vals_for_plot
    mean_abs_shap = np.mean(np.abs(raw_shap_matrix), axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_features = [X_sample.columns[i] for i in top_feature_indices]

    logger.info(f"Top 5 most impactful features by mean(|SHAP|): {top_features}")

    return {
        "summary_plot": str(summary_plot_path),
        "tp_plot": str(tp_plot_path) if len(tp_indices) > 0 else None,
        "fp_plot": str(fp_plot_path),
        "top_features": top_features,
    }


if __name__ == "__main__":
    run_shap_analysis()
