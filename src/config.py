"""
Central configuration module for the Credit Card Fraud Detection Pipeline.
All paths, hyperparameters, column schemas, and environmental variables are defined here.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_PLOTS_DIR = MODELS_DIR / "eval_plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure essential output directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Saved Artifact File Paths
MODEL_ARTIFACT_PATH = MODELS_DIR / "fraud_detector.joblib"
SCALER_ARTIFACT_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Dataset Column Definitions
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURE_COLS = [TIME_COL] + V_COLS + [AMOUNT_COL]

# Engineered Feature Column Names
AMOUNT_LOG_COL = "amount_log"
HOUR_SIN_COL = "hour_sin"
HOUR_COS_COL = "hour_cos"
TX_VELOCITY_COL = "tx_velocity_1h"

ALL_ENGINEERED_COLS = [AMOUNT_LOG_COL, HOUR_SIN_COL, HOUR_COS_COL, TX_VELOCITY_COL]

# Splitting & Random Seed
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

# Kaggle Dataset Identifier
KAGGLE_DATASET_ID = "mlg-ulb/creditcardfraud"
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# MLflow Experiment Tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
MLFLOW_DB_PATH = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Credit_Card_Fraud_Detection")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")

# API Serving
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_FILE_PATH = LOGS_DIR / "pipeline.log"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.log"

# Default XGBoost & Imbalance Hyperparameters
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1.0, 5.0, 10.0, 50.0, 100.0, 580.0],
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
}
