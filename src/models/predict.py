"""
Inference prediction engine for Credit Card Fraud Detection.

Loads serialized model, scaler, and metadata once and serves single or batch predictions
with full feature engineering and scaling transformations.
"""

from typing import Dict, Any, List, Union
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_ARTIFACT_PATH,
    SCALER_ARTIFACT_PATH,
    FEATURE_NAMES_PATH,
    TARGET_COL,
)
from src.data.preprocess import FraudDataPreprocessor
from src.features.build_features import (
    engineer_single_record,
    engineer_features,
    get_feature_column_order,
)

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    High-performance inference engine for fraud scoring.
    """

    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_PATH,
        scaler_path: Path = SCALER_ARTIFACT_PATH,
        threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.threshold = threshold
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_version = "1.0.0"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model, scaler, and feature definitions."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}. Train the model first.")
        
        logger.info(f"Loading production model from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        self.preprocessor = FraudDataPreprocessor(self.scaler_path)
        self.preprocessor.load()

        if FEATURE_NAMES_PATH.exists():
            self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        else:
            self.feature_names = get_feature_column_order()

        logger.info("Inference engine initialized successfully.")

    def predict_single(self, transaction: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
        """
        Score a single raw transaction dictionary.

        Args:
            transaction: Dictionary containing 'Time', 'Amount', and 'V1'..'V28'.
            threshold: Optional custom decision threshold.

        Returns:
            Dict[str, Any]: Prediction result containing fraud_probability and fraud_flag.
        """
        th = threshold if threshold is not None else self.threshold

        # Step 1: Feature Engineering
        enriched = engineer_single_record(transaction)
        
        # Step 2: Format into single-row DataFrame
        df_row = pd.DataFrame([enriched])
        
        # Step 3: Scale
        scaled_row = self.preprocessor.transform(df_row)

        # Step 4: Model inference
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(scaled_row)[0, 1])
        else:
            prob = float(self.model.predict(scaled_row)[0])

        flag = bool(prob >= th)

        return {
            "fraud_probability": round(prob, 4),
            "fraud_flag": flag,
            "decision_threshold": th,
            "model_version": self.model_version,
        }

    def predict_batch(self, transactions: List[Dict[str, Any]], threshold: float = None) -> List[Dict[str, Any]]:
        """
        Score a batch of transactions efficiently.

        Args:
            transactions: List of transaction dictionaries.
            threshold: Optional custom decision threshold.

        Returns:
            List[Dict[str, Any]]: List of prediction results.
        """
        if not transactions:
            return []

        th = threshold if threshold is not None else self.threshold

        df_raw = pd.DataFrame(transactions)
        df_engineered = engineer_features(df_raw, is_sorted_stream=True)
        df_scaled = self.preprocessor.transform(df_engineered)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_scaled)[:, 1]
        else:
            probs = self.model.predict(df_scaled)

        results = []
        for p in probs:
            prob = float(p)
            results.append({
                "fraud_probability": round(prob, 4),
                "fraud_flag": bool(prob >= th),
                "decision_threshold": th,
                "model_version": self.model_version,
            })

        return results


# Global singleton instance for FastAPI serving
_predictor_instance = None


def get_predictor() -> FraudPredictor:
    """Return singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor()
    return _predictor_instance
