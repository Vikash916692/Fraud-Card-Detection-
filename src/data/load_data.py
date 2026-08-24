"""
Data ingestion module for Credit Card Fraud Detection.
Loads raw transaction dataset, validates schema and column types, and logs initial summary stats.
"""

from pathlib import Path
from typing import Optional
import logging
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    RAW_FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    AMOUNT_COL,
    V_COLS,
)

logger = logging.getLogger(__name__)


def get_expected_dtypes() -> dict:
    """Return dictionary mapping column names to explicit data types."""
    dtypes = {TIME_COL: np.float64, AMOUNT_COL: np.float64}
    for col in V_COLS:
        dtypes[col] = np.float64
    dtypes[TARGET_COL] = np.int32
    return dtypes


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw credit card transaction data from CSV file.

    Args:
        file_path: Path to creditcard.csv. Defaults to RAW_DATA_FILE from config.

    Returns:
        pd.DataFrame: Validated raw transaction dataframe.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing or data cannot be parsed.
    """
    target_path = Path(file_path) if file_path is not None else RAW_DATA_FILE

    if not target_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{target_path}'. "
            "Please run 'python scripts/download_data.py' or place creditcard.csv in data/raw/."
        )

    logger.info(f"Loading raw dataset from {target_path}...")
    expected_dtypes = get_expected_dtypes()

    try:
        # Load CSV with explicit dtypes
        df = pd.read_csv(target_path, dtype=expected_dtypes)
    except Exception as e:
        logger.warning(f"Could not load with strict dtypes directly ({e}). Attempting fallback load...")
        df = pd.read_csv(target_path)
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    # Validate required columns
    required_cols = set(RAW_FEATURE_COLS + [TARGET_COL])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Check for missing values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        logger.warning(f"Dataset contains {null_counts} missing values. Imputing with median...")
        df = df.fillna(df.median())

    total_rows = len(df)
    fraud_rows = int(df[TARGET_COL].sum()) if TARGET_COL in df.columns else 0
    fraud_pct = (fraud_rows / total_rows * 100) if total_rows > 0 else 0.0

    logger.info(
        f"Successfully loaded dataset. Shape: {df.shape} | "
        f"Total: {total_rows:,} | Frauds: {fraud_rows:,} ({fraud_pct:.3f}%) | "
        f"Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        data = load_raw_data()
        print("First 5 rows:")
        print(data.head())
    except Exception as exc:
        print(f"Error: {exc}")
