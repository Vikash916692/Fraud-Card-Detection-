"""
Unit tests for feature engineering logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    compute_cyclical_time_features,
    compute_log_amount_feature,
    compute_stream_velocity_feature,
    engineer_features,
    engineer_single_record,
    get_feature_column_order,
)
from src.config import (
    TIME_COL,
    AMOUNT_COL,
    AMOUNT_LOG_COL,
    HOUR_SIN_COL,
    HOUR_COS_COL,
    TX_VELOCITY_COL,
)


@pytest.fixture
def sample_raw_df():
    """Create synthetic raw transaction DataFrame."""
    np.random.seed(42)
    n = 100
    times = np.sort(np.random.uniform(0, 86400 * 2, n))  # 2 days in seconds
    amounts = np.random.exponential(scale=50.0, size=n)
    
    data = {TIME_COL: times, AMOUNT_COL: amounts}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    data["Class"] = (np.random.rand(n) < 0.05).astype(int)

    return pd.DataFrame(data)


def test_cyclical_time_features(sample_raw_df):
    """Test sine and cosine conversions of timestamps."""
    df_out = compute_cyclical_time_features(sample_raw_df)
    
    assert HOUR_SIN_COL in df_out.columns
    assert HOUR_COS_COL in df_out.columns

    # Sin and Cos must stay bounded in [-1, 1]
    assert df_out[HOUR_SIN_COL].between(-1.0001, 1.0001).all()
    assert df_out[HOUR_COS_COL].between(-1.0001, 1.0001).all()

    # Verify sin^2 + cos^2 ≈ 1
    sin_sq_plus_cos_sq = df_out[HOUR_SIN_COL] ** 2 + df_out[HOUR_COS_COL] ** 2
    assert np.allclose(sin_sq_plus_cos_sq, 1.0, atol=1e-5)


def test_log_amount_feature(sample_raw_df):
    """Test log1p transformation of transaction amounts."""
    df_out = compute_log_amount_feature(sample_raw_df)
    
    assert AMOUNT_LOG_COL in df_out.columns
    assert (df_out[AMOUNT_LOG_COL] >= 0.0).all()

    # Zero amount test
    zero_df = pd.DataFrame({AMOUNT_COL: [0.0]})
    zero_out = compute_log_amount_feature(zero_df)
    assert zero_out[AMOUNT_LOG_COL].iloc[0] == 0.0


def test_stream_velocity_feature(sample_raw_df):
    """Test rolling window velocity calculation."""
    df_out = compute_stream_velocity_feature(sample_raw_df, window_seconds=3600.0)
    
    assert TX_VELOCITY_COL in df_out.columns
    assert (df_out[TX_VELOCITY_COL] >= 1.0).all()
    # Velocity at index 0 must be 1
    assert df_out[TX_VELOCITY_COL].iloc[0] == 1.0


def test_full_engineer_features(sample_raw_df):
    """Test complete feature engineering pipeline."""
    df_out = engineer_features(sample_raw_df)
    
    expected_cols = [AMOUNT_LOG_COL, HOUR_SIN_COL, HOUR_COS_COL, TX_VELOCITY_COL]
    for col in expected_cols:
        assert col in df_out.columns

    assert len(df_out) == len(sample_raw_df)


def test_engineer_single_record():
    """Test single record transformation for API serving."""
    raw_record = {"Time": 7200.0, "Amount": 100.0}
    for i in range(1, 29):
        raw_record[f"V{i}"] = 0.5

    enriched = engineer_single_record(raw_record)

    assert HOUR_SIN_COL in enriched
    assert HOUR_COS_COL in enriched
    assert AMOUNT_LOG_COL in enriched
    assert TX_VELOCITY_COL in enriched
    assert np.isclose(enriched[AMOUNT_LOG_COL], np.log1p(100.0))
