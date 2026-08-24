"""
Unit tests for data preprocessing and leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FraudDataPreprocessor, prepare_train_test_data
from src.config import RANDOM_STATE, TIME_COL, AMOUNT_COL, TARGET_COL, TEST_SIZE


@pytest.fixture
def mock_dataset():
    """Create a controlled mock dataset for testing splits and scalers."""
    np.random.seed(RANDOM_STATE)
    n = 1000
    n_fraud = 10  # 1% fraud

    times = np.linspace(0, 10000, n)
    amounts = np.random.uniform(5, 500, n)
    data = {TIME_COL: times, AMOUNT_COL: amounts}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)

    labels = np.zeros(n, dtype=int)
    fraud_indices = np.random.choice(n, size=n_fraud, replace=False)
    labels[fraud_indices] = 1
    data[TARGET_COL] = labels

    return pd.DataFrame(data)


def test_preprocessor_fitting_and_transformation(mock_dataset):
    """Ensure preprocessor fits on training data and transforms without throwing errors."""
    from src.features.build_features import engineer_features

    df_eng = engineer_features(mock_dataset)
    X = df_eng.drop(columns=[TARGET_COL])

    preprocessor = FraudDataPreprocessor()
    assert not preprocessor.is_fitted

    X_scaled = preprocessor.fit_transform(X)
    assert preprocessor.is_fitted
    assert X_scaled.shape[0] == X.shape[0]
    assert X_scaled.shape[1] == len(preprocessor.feature_columns)


def test_no_data_leakage_in_scaler():
    """Verify that test set statistics do not influence the scaler."""
    train_data = pd.DataFrame({
        TIME_COL: [100.0, 200.0, 300.0, 400.0],
        AMOUNT_COL: [10.0, 20.0, 30.0, 40.0],
        "amount_log": [np.log1p(10), np.log1p(20), np.log1p(30), np.log1p(40)],
        "tx_velocity_1h": [1.0, 2.0, 3.0, 4.0],
    })
    for i in range(1, 29):
        train_data[f"V{i}"] = 0.0

    test_data = pd.DataFrame({
        TIME_COL: [100000.0],  # Outlier in test set
        AMOUNT_COL: [99999.0],  # Outlier in test set
        "amount_log": [np.log1p(99999)],
        "tx_velocity_1h": [500.0],
    })
    for i in range(1, 29):
        test_data[f"V{i}"] = 0.0

    preprocessor = FraudDataPreprocessor()
    preprocessor.fit(train_data)

    # Median and scale should reflect train_data ONLY
    center_amount = preprocessor.scaler.center_[1]  # index 1 is AMOUNT_COL
    assert np.isclose(center_amount, 25.0)  # Median of [10, 20, 30, 40] is 25.0


def test_stratified_split_preserves_ratio(mock_dataset):
    """Test that train/test split maintains positive class ratio accurately."""
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data(
        raw_df=mock_dataset, save_processed=False
    )

    orig_ratio = mock_dataset[TARGET_COL].mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert np.isclose(train_ratio, orig_ratio, atol=0.01)
    assert np.isclose(test_ratio, orig_ratio, atol=0.01)
    assert len(X_test) == int(len(mock_dataset) * TEST_SIZE)
