"""
Dataset download and validation script for Credit Card Fraud Detection.

Supports:
1. Kaggle API download (using KAGGLE_USERNAME & KAGGLE_KEY from .env or ~/.kaggle/kaggle.json).
2. Direct local verification if creditcard.csv is placed in data/raw/.
3. Explicitly gated --synthetic flag for CI/unit testing only (creates SYNTHETIC_DATA_USED.flag).
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure parent directory is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    KAGGLE_DATASET_ID,
    KAGGLE_USERNAME,
    KAGGLE_KEY,
    RANDOM_STATE,
    RAW_FEATURE_COLS,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("download_data")


def verify_dataset(file_path: Path) -> bool:
    """Verify that creditcard.csv exists and matches expected schema."""
    if not file_path.exists():
        return False
    
    logger.info(f"Validating dataset at: {file_path}")
    try:
        sample_df = pd.read_csv(file_path, nrows=10)
        expected_cols = set(RAW_FEATURE_COLS + [TARGET_COL])
        actual_cols = set(sample_df.columns)
        
        if not expected_cols.issubset(actual_cols):
            logger.error(f"Missing required columns! Expected {expected_cols}, got {actual_cols}")
            return False
            
        logger.info("Dataset validation successful! Schema matches expected ULB Credit Card Fraud dataset.")
        return True
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return False


def download_from_kaggle() -> bool:
    """Download credit card fraud dataset using Kaggle API."""
    logger.info("Attempting to download dataset from Kaggle...")

    # Set environment variables for Kaggle API if present in .env
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY

    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    has_creds = (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or kaggle_json_path.exists()

    if not has_creds:
        logger.warning("No Kaggle API credentials found in environment (.env) or ~/.kaggle/kaggle.json.")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Authenticated with Kaggle API. Downloading '{KAGGLE_DATASET_ID}'...")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET_ID, path=str(RAW_DATA_DIR), unzip=True)
        
        if verify_dataset(RAW_DATA_FILE):
            logger.info(f"Dataset successfully downloaded and verified at {RAW_DATA_FILE}")
            return True
    except Exception as e:
        logger.error(f"Kaggle API download failed: {e}")

    return False


def generate_synthetic_dataset(output_path: Path, n_samples: int = 50000) -> None:
    """
    Generate synthetic dataset matching the ULB schema for testing/CI environments only.
    Writes a SYNTHETIC_DATA_USED.flag file to prevent accidental reporting of synthetic metrics.
    """
    logger.warning("=" * 70)
    logger.warning("GENERATING SYNTHETIC DATASET (--synthetic flag provided).")
    logger.warning("THIS IS STRICTLY FOR CI / TESTING PURPOSES.")
    logger.warning("=" * 70)

    np.random.seed(RANDOM_STATE)
    n_fraud = int(n_samples * 0.00172)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_time = np.sort(np.random.uniform(0, 172800, n_legit))
    legit_amount = np.random.exponential(scale=88.0, size=n_legit)
    legit_v = np.random.randn(n_legit, 28)

    # Fraudulent transactions (different mean/variance in PCA space & skewed amounts)
    fraud_time = np.sort(np.random.uniform(0, 172800, n_fraud))
    fraud_amount = np.random.exponential(scale=120.0, size=n_fraud)
    fraud_v = np.random.randn(n_fraud, 28)
    fraud_v[:, 13] -= 2.5  # Simulate V14 drop
    fraud_v[:, 16] -= 2.0  # Simulate V17 drop
    fraud_v[:, 11] += 2.0  # Simulate V12 shift

    # Construct DataFrame
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    
    legit_data = np.column_stack([legit_time, legit_v, legit_amount, np.zeros(n_legit)])
    fraud_data = np.column_stack([fraud_time, fraud_v, fraud_amount, np.ones(n_fraud)])
    
    all_data = np.vstack([legit_data, fraud_data])
    df = pd.DataFrame(all_data, columns=cols)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    flag_file = output_path.parent / "SYNTHETIC_DATA_USED.flag"
    with open(flag_file, "w") as f:
        f.write("SYNTHETIC DATA USED FOR TESTING/CI ONLY.\nDo not quote in interview/benchmarks.\n")

    logger.info(f"Synthetic dataset saved to {output_path} (n={len(df)}, frauds={df['Class'].sum()})")
    logger.info(f"Created warning flag at {flag_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and verify credit card fraud dataset.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic dataset for testing/CI only. Defaults to False.",
    )
    args = parser.parse_args()

    # 1. Check if real dataset already exists
    if RAW_DATA_FILE.exists():
        if verify_dataset(RAW_DATA_FILE):
            logger.info(f"Real dataset already exists at {RAW_DATA_FILE}. Ready for training.")
            return

    # 2. If --synthetic flag is provided, generate synthetic data for testing
    if args.synthetic:
        generate_synthetic_dataset(RAW_DATA_FILE)
        return

    # 3. Attempt Kaggle download
    if download_from_kaggle():
        return

    # 4. If all fail, exit with clear instructions
    logger.error("=" * 70)
    logger.error("DATASET NOT FOUND AND DOWNLOAD FAILED")
    logger.error("=" * 70)
    logger.error(
        "To obtain the Kaggle 'Credit Card Fraud Detection' dataset:\n"
        "1. Option A (Automated): Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file,\n"
        "   or place your kaggle.json in ~/.kaggle/kaggle.json.\n"
        "   (Obtain credentials from Kaggle -> Account Settings -> API -> Create New Token)\n\n"
        "2. Option B (Manual): Download 'creditcard.csv' directly from:\n"
        "   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        f"   and place it at: {RAW_DATA_FILE.resolve()}\n\n"
        "3. Option C (CI/Unit Testing Only): Pass the --synthetic flag:\n"
        "   python scripts/download_data.py --synthetic\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
