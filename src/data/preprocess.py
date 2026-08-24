"""
Data preprocessing and splitting module for Credit Card Fraud Detection.

Handles:
1. Feature extraction and ordering.
2. Stratified train/test split (80/20) with fixed random seed.
3. Fitting RobustScaler strictly on the training set (preventing data leakage).
4. Persisting fitted scaler and processed datasets to disk for training & inference.
"""

from typing import Tuple, List
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER_ARTIFACT_PATH,
    PROCESSED_DATA_DIR,
    FEATURE_NAMES_PATH,
    TIME_COL,
    AMOUNT_COL,
    AMOUNT_LOG_COL,
    TX_VELOCITY_COL,
)
from src.data.load_data import load_raw_data
from src.features.build_features import engineer_features, get_feature_column_order

logger = logging.getLogger(__name__)

# Features with high skew / outliers that require RobustScaler
COLUMNS_TO_SCALE = [TIME_COL, AMOUNT_COL, AMOUNT_LOG_COL, TX_VELOCITY_COL]


class FraudDataPreprocessor:
    """
    Preprocessor class that manages scaling transformations without data leakage.
    """

    def __init__(self, scaler_path: Path = SCALER_ARTIFACT_PATH):
        self.scaler_path = Path(scaler_path)
        self.scaler = RobustScaler()
        self.columns_to_scale = COLUMNS_TO_SCALE
        self.feature_columns: List[str] = get_feature_column_order()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "FraudDataPreprocessor":
        """
        Fit RobustScaler strictly on training data.
        """
        logger.info(f"Fitting RobustScaler on columns: {self.columns_to_scale}")
        cols_present = [c for c in self.columns_to_scale if c in X_train.columns]
        self.scaler.fit(X_train[cols_present])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset using fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        X_out = X.copy()
        cols_present = [c for c in self.columns_to_scale if c in X_out.columns]
        X_out[cols_present] = self.scaler.transform(X_out[cols_present])
        
        # Ensure standard column order
        ordered_cols = [c for c in self.feature_columns if c in X_out.columns]
        return X_out[ordered_cols]

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and transform it."""
        return self.fit(X_train).transform(X_train)

    def save(self, path: Path = None) -> None:
        """Persist fitted scaler to disk."""
        target_path = Path(path) if path else self.scaler_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, target_path)
        # Also persist feature names
        joblib.dump(self.feature_columns, FEATURE_NAMES_PATH)
        logger.info(f"Saved fitted scaler to {target_path} and feature names to {FEATURE_NAMES_PATH}")

    def load(self, path: Path = None) -> "FraudDataPreprocessor":
        """Load pre-fitted scaler from disk."""
        target_path = Path(path) if path else self.scaler_path
        if not target_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {target_path}")
        self.scaler = joblib.load(target_path)
        if FEATURE_NAMES_PATH.exists():
            self.feature_columns = joblib.load(FEATURE_NAMES_PATH)
        self.is_fitted = True
        logger.info(f"Loaded scaler from {target_path}")
        return self


def prepare_train_test_data(
    raw_df: pd.DataFrame = None,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FraudDataPreprocessor]:
    """
    Complete end-to-end preprocessing workflow:
    1. Ingest raw data (if not provided).
    2. Run feature engineering.
    3. Perform Stratified Train/Test split.
    4. Fit RobustScaler strictly on X_train, then transform both X_train and X_test.
    5. Save processed artifacts.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, preprocessor)
    """
    if raw_df is None:
        raw_df = load_raw_data()

    # Step 1: Feature Engineering
    engineered_df = engineer_features(raw_df)

    # Step 2: Separate features and target
    feature_cols = get_feature_column_order()
    X = engineered_df[feature_cols]
    y = engineered_df[TARGET_COL]

    logger.info(f"Splitting dataset: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_fraud_rate = (y_train.sum() / len(y_train)) * 100
    test_fraud_rate = (y_test.sum() / len(y_test)) * 100
    logger.info(
        f"Train set: {len(X_train):,} rows ({y_train.sum()} frauds, {train_fraud_rate:.3f}%) | "
        f"Test set: {len(X_test):,} rows ({y_test.sum()} frauds, {test_fraud_rate:.3f}%)"
    )

    # Step 3: Scaling without data leakage
    preprocessor = FraudDataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Step 4: Persist artifacts
    if save_processed:
        preprocessor.save()
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            X_train_scaled.to_parquet(PROCESSED_DATA_DIR / "X_train.parquet", index=False)
            X_test_scaled.to_parquet(PROCESSED_DATA_DIR / "X_test.parquet", index=False)
            y_train.to_frame().to_parquet(PROCESSED_DATA_DIR / "y_train.parquet", index=False)
            y_test.to_frame().to_parquet(PROCESSED_DATA_DIR / "y_test.parquet", index=False)
        except Exception:
            # Fallback to CSV if parquet engine not installed
            X_train_scaled.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
            X_test_scaled.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
            y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
            y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
        logger.info(f"Saved processed data splits to {PROCESSED_DATA_DIR}")

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_train_test_data()
