"""
Model training, hyperparameter optimization, and MLflow experiment tracking module.

Trains and compares:
1. Baseline: Logistic Regression (class_weight='balanced')
2. Primary Model: XGBoost + scale_pos_weight (with 5-fold StratifiedKFold CV scored on AUC-PR)
3. Comparison Model: XGBoost + SMOTE (minority oversampling on train split only)

Selects the production model based on highest test AUC-PR and persists artifacts.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, Any, Tuple
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    MODEL_ARTIFACT_PATH,
    MODEL_METRICS_PATH,
    RANDOM_STATE,
    CV_SPLITS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    XGB_PARAM_GRID,
    LOGISTIC_REGRESSION_PARAMS,
    EVAL_PLOTS_DIR,
)
from src.data.preprocess import prepare_train_test_data
from src.models.evaluate import evaluate_model, plot_evaluation_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("model_training")


def setup_mlflow():
    """Configure MLflow tracking store and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow configured with tracking URI: {MLFLOW_TRACKING_URI}")


def train_logistic_regression_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train a Logistic Regression baseline model with balanced class weights.
    """
    logger.info("=" * 60)
    logger.info("Training Model 1: Logistic Regression Baseline (class_weight='balanced')...")
    logger.info("=" * 60)

    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, model_name="Logistic_Regression_Baseline", generate_plots=False)

    try:
        with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
            mlflow.log_params(LOGISTIC_REGRESSION_PARAMS)
            for metric_name, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", val)
    except Exception as e:
        logger.warning(f"MLflow logging notice: {e}")

    return model, metrics


def tune_and_train_xgboost_weighted(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 8,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Tune and train XGBoost using class weighting (scale_pos_weight) and 5-Fold StratifiedKFold CV
    scored strictly on average_precision (AUC-PR).
    """
    logger.info("=" * 60)
    logger.info("Training Model 2: XGBoost + scale_pos_weight (StratifiedKFold CV on AUC-PR)...")
    logger.info("=" * 60)

    # Ratio of negative to positive class in training data (~578)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    natural_scale_pos_weight = float(neg_count / max(pos_count, 1))

    # Candidate scale_pos_weight values
    param_distributions = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1.0, 5.0, 10.0, 50.0, natural_scale_pos_weight],
    }

    base_xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",  # AUC-PR optimization
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=1,  # Single-process dispatch with XGBoost internal multi-threading for speed & stability
        refit=True,
    )

    logger.info(f"Running RandomizedSearchCV ({n_iter} iterations, {CV_SPLITS}-fold CV scored on AUC-PR)...")
    search.fit(X_train, y_train)

    best_model: XGBClassifier = search.best_estimator_
    logger.info(f"Best CV Average Precision (AUC-PR): {search.best_score_:.4f}")
    logger.info(f"Best Hyperparameters: {search.best_params_}")

    metrics = evaluate_model(best_model, X_test, y_test, model_name="XGBoost_Class_Weighted", generate_plots=False)
    metrics["best_cv_auc_pr"] = float(search.best_score_)

    try:
        with mlflow.start_run(run_name="XGBoost_Class_Weighted"):
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_best_auc_pr", float(search.best_score_))
            for metric_name, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", val)
    except Exception as e:
        logger.warning(f"MLflow logging notice: {e}")

    return best_model, metrics


def train_xgboost_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict[str, Any],
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train XGBoost with SMOTE oversampling applied strictly to the training data.
    """
    logger.info("=" * 60)
    logger.info("Training Model 3: XGBoost + SMOTE (minority oversampling on train split only)...")
    logger.info("=" * 60)

    # SMOTE applied strictly to train set to avoid any data leakage
    # Target 10:1 ratio (sampling_strategy=0.1) to avoid synthesizing excessive noise
    smote = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
    logger.info("Applying SMOTE to training split...")
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        f"Post-SMOTE train shape: {X_train_resampled.shape} | "
        f"Fraud samples: {y_train_resampled.sum():,} (from original {y_train.sum():,})"
    )

    # Use tuned tree params but reset scale_pos_weight to 1.0 since SMOTE balanced the distribution
    smote_params = dict(best_params)
    smote_params["scale_pos_weight"] = 1.0
    smote_params["random_state"] = RANDOM_STATE
    smote_params["eval_metric"] = "aucpr"
    smote_params["tree_method"] = "hist"
    smote_params["n_jobs"] = -1

    model = XGBClassifier(**smote_params)
    model.fit(X_train_resampled, y_train_resampled)

    metrics = evaluate_model(model, X_test, y_test, model_name="XGBoost_SMOTE", generate_plots=False)

    try:
        with mlflow.start_run(run_name="XGBoost_SMOTE"):
            mlflow.log_params(smote_params)
            mlflow.log_param("smote_sampling_strategy", 0.1)
            for metric_name, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", val)
    except Exception as e:
        logger.warning(f"MLflow logging notice: {e}")

    return model, metrics


def run_training_pipeline(n_tuning_iter: int = 8) -> Dict[str, Any]:
    """
    Execute full multi-model training, MLflow tracking, and model selection.
    """
    setup_mlflow()

    # Step 1: Preprocess and split
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data()

    # Step 2: Model 1 - Logistic Regression Baseline
    lr_model, lr_metrics = train_logistic_regression_baseline(X_train, y_train, X_test, y_test)

    # Step 3: Model 2 - XGBoost + Class Weighting (with 5-fold StratifiedKFold CV on AUC-PR)
    xgb_weighted_model, xgb_weighted_metrics = tune_and_train_xgboost_weighted(
        X_train, y_train, X_test, y_test, n_iter=n_tuning_iter
    )

    # Extract best tuned params
    best_tree_params = {
        "n_estimators": int(xgb_weighted_model.n_estimators),
        "max_depth": int(xgb_weighted_model.max_depth),
        "learning_rate": float(xgb_weighted_model.learning_rate),
        "subsample": float(xgb_weighted_model.subsample),
        "colsample_bytree": float(xgb_weighted_model.colsample_bytree),
    }

    # Step 4: Model 3 - XGBoost + SMOTE
    xgb_smote_model, xgb_smote_metrics = train_xgboost_smote(
        X_train, y_train, X_test, y_test, best_tree_params
    )

    # Step 5: Model Comparison & Selection (Explicitly based on Test AUC-PR)
    all_results = {
        "Logistic_Regression_Baseline": {"model": lr_model, "metrics": lr_metrics},
        "XGBoost_Class_Weighted": {"model": xgb_weighted_model, "metrics": xgb_weighted_metrics},
        "XGBoost_SMOTE": {"model": xgb_smote_model, "metrics": xgb_smote_metrics},
    }

    logger.info("=" * 70)
    logger.info("MODEL COMPARISON SUMMARY (Ranked by Test AUC-PR):")
    logger.info("=" * 70)
    
    sorted_models = sorted(
        all_results.items(),
        key=lambda item: item[1]["metrics"]["auc_pr"],
        reverse=True,
    )

    for name, item in sorted_models:
        m = item["metrics"]
        logger.info(
            f"Model: {name:<28} | AUC-PR: {m['auc_pr']:.4f} | "
            f"AUC-ROC: {m['auc_roc']:.4f} | F1: {m['f1_score']:.4f} | "
            f"Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f} | KS: {m['ks_statistic']:.4f}"
        )

    best_model_name, best_info = sorted_models[0]
    best_model = best_info["model"]
    best_metrics = best_info["metrics"]

    logger.info("=" * 70)
    logger.info(f"SELECTED PRODUCTION MODEL: {best_model_name} (Test AUC-PR: {best_metrics['auc_pr']:.4f})")
    logger.info("=" * 70)

    # Persist production model
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_ARTIFACT_PATH)
    logger.info(f"Saved selected production model to: {MODEL_ARTIFACT_PATH}")

    # Generate and save final evaluation plots for the winning model
    y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(X_test)
    plot_evaluation_curves(y_test.values, y_test_proba, model_name=best_model_name, output_dir=EVAL_PLOTS_DIR)

    # Save metrics JSON
    metrics_summary = {
        "selected_model": best_model_name,
        "selection_criterion": "Highest Test AUC-PR (Precision-Recall AUC under severe class imbalance)",
        "models": {name: item["metrics"] for name, item in all_results.items()},
    }
    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved complete metrics summary to: {MODEL_METRICS_PATH}")

    return metrics_summary


if __name__ == "__main__":
    run_training_pipeline()
