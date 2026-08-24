"""
Feature engineering module for the Credit Card Fraud Detection Pipeline.

Engineered Features:
1. hour_sin & hour_cos:
   - Rationale: Time is provided as seconds elapsed from the dataset start. Converting Time into
     an hour of day (0-23) and projecting it onto continuous sine and cosine waves captures
     the 24-hour diurnal cycle of human behavior without creating artificial boundaries (e.g.,
     23:59 and 00:00 are close in cyclical space).
   - Formula:
       hour = (Time // 3600) % 24
       hour_sin = sin(2 * pi * hour / 24)
       hour_cos = cos(2 * pi * hour / 24)

2. amount_log:
   - Rationale: Financial transaction amounts are highly right-skewed with extreme outliers
     (e.g., from $0 to $25,000+). Log1p transform (log(1 + x)) stabilizes variance and compresses
     the heavy right tail while gracefully handling $0 authorization probes.
   - Formula:
       amount_log = log(1 + Amount)

3. tx_velocity_1h:
   - Rationale: Fraudsters often execute rapid burst attacks. Because the Kaggle dataset is anonymized
     without explicit customer or card IDs, this feature calculates the aggregate burstiness /
     transaction arrival velocity over a 1-hour trailing window (3,600 seconds) in the global stream.
     (Note: In real-world enterprise banking, this would be computed per Cardholder/Account ID).
"""

from typing import Dict, Any, List
import logging
import numpy as np
import pandas as pd

from src.config import (
    TIME_COL,
    AMOUNT_COL,
    AMOUNT_LOG_COL,
    HOUR_SIN_COL,
    HOUR_COS_COL,
    TX_VELOCITY_COL,
    V_COLS,
)

logger = logging.getLogger(__name__)


def compute_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform transaction timestamp (seconds) into 24-hour diurnal sine and cosine features.

    Args:
        df: DataFrame containing 'Time' column.

    Returns:
        pd.DataFrame: DataFrame with 'hour_sin' and 'hour_cos' columns added.
    """
    df = df.copy()
    hours = (df[TIME_COL] // 3600) % 24
    radians = 2 * np.pi * hours / 24.0
    df[HOUR_SIN_COL] = np.sin(radians)
    df[HOUR_COS_COL] = np.cos(radians)
    return df


def compute_log_amount_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply natural logarithm log(1 + x) transformation to transaction Amount.

    Args:
        df: DataFrame containing 'Amount' column.

    Returns:
        pd.DataFrame: DataFrame with 'amount_log' column added.
    """
    df = df.copy()
    # Clip negative amounts if any to zero before log1p
    clipped_amount = np.maximum(df[AMOUNT_COL].values, 0.0)
    df[AMOUNT_LOG_COL] = np.log1p(clipped_amount)
    return df


def compute_stream_velocity_feature(df: pd.DataFrame, window_seconds: float = 3600.0) -> pd.DataFrame:
    """
    Compute transaction velocity (count of transactions in a trailing time window).

    Args:
        df: DataFrame sorted by Time.
        window_seconds: Trailing time window in seconds (default: 3600s = 1 hour).

    Returns:
        pd.DataFrame: DataFrame with 'tx_velocity_1h' feature.
    """
    df = df.copy()
    times = df[TIME_COL].values
    
    # Efficient rolling search over sorted timestamps
    # For large datasets, np.searchsorted gives O(N log N) performance
    left_indices = np.searchsorted(times, times - window_seconds, side="left")
    right_indices = np.arange(len(times)) + 1
    counts = right_indices - left_indices

    df[TX_VELOCITY_COL] = counts.astype(np.float64)
    return df


def engineer_features(df: pd.DataFrame, is_sorted_stream: bool = True) -> pd.DataFrame:
    """
    Execute complete feature engineering pipeline on a DataFrame.

    Args:
        df: Raw DataFrame containing 'Time', 'Amount', and 'V1'..'V28'.
        is_sorted_stream: Whether the DataFrame is in chronological order.

    Returns:
        pd.DataFrame: DataFrame augmented with all engineered features.
    """
    logger.info("Executing feature engineering pipeline...")
    
    df_out = df.copy()
    
    # 1. Cyclical time features
    df_out = compute_cyclical_time_features(df_out)
    
    # 2. Log amount feature
    df_out = compute_log_amount_feature(df_out)
    
    # 3. Trailing velocity feature
    if is_sorted_stream and len(df_out) > 1:
        df_out = compute_stream_velocity_feature(df_out)
    else:
        # Default single transaction or unsorted stream velocity
        df_out[TX_VELOCITY_COL] = 1.0

    logger.info(f"Feature engineering complete. Output columns count: {len(df_out.columns)}")
    return df_out


def engineer_single_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply feature engineering transformations to a single transaction payload dictionary for API inference.

    Args:
        record: Dictionary with raw transaction fields (Time, Amount, V1-V28).

    Returns:
        Dict[str, Any]: Dictionary enriched with engineered features.
    """
    time_val = float(record.get(TIME_COL, 0.0))
    amount_val = float(record.get(AMOUNT_COL, 0.0))

    # Cyclical hour
    hour = (time_val // 3600) % 24
    radians = 2 * np.pi * hour / 24.0
    hour_sin = float(np.sin(radians))
    hour_cos = float(np.cos(radians))

    # Log amount
    amount_log = float(np.log1p(max(amount_val, 0.0)))

    # Stream velocity fallback for single isolated request
    velocity = float(record.get(TX_VELOCITY_COL, 1.0))

    enriched = dict(record)
    enriched[HOUR_SIN_COL] = hour_sin
    enriched[HOUR_COS_COL] = hour_cos
    enriched[AMOUNT_LOG_COL] = amount_log
    enriched[TX_VELOCITY_COL] = velocity

    return enriched


def get_feature_column_order() -> List[str]:
    """
    Return standard ordered list of all feature names passed to the model.
    """
    return (
        [TIME_COL]
        + V_COLS
        + [AMOUNT_COL]
        + [AMOUNT_LOG_COL, HOUR_SIN_COL, HOUR_COS_COL, TX_VELOCITY_COL]
    )
