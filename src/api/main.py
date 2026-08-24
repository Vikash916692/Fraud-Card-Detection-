"""
FastAPI REST API application for real-time Credit Card Fraud Detection.

Endpoints:
- POST /predict: Score individual transaction payload
- POST /predict/batch: Score multiple transactions in batch
- GET /health: Service and model operational status
- GET /: API metadata and documentation link
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    TransactionInput,
    BatchTransactionInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from src.config import PREDICTIONS_LOG_PATH, LOGS_DIR
from src.models.predict import get_predictor, FraudPredictor

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Dedicated structured logger for predictions
pred_logger = logging.getLogger("fraud_predictions")
pred_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(PREDICTIONS_LOG_PATH, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))
if not pred_logger.handlers:
    pred_logger.addHandler(file_handler)

logger = logging.getLogger("api_main")
start_time = time.time()
predictor_instance: Optional[FraudPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads ML artifacts once at startup."""
    global predictor_instance
    logger.info("Initializing Fraud Predictor artifacts at startup...")
    try:
        predictor_instance = get_predictor()
        logger.info("Model and preprocessor successfully loaded into memory.")
    except Exception as e:
        logger.warning(f"Could not preload model at startup ({e}). Will load on demand.")
    yield
    logger.info("Shutting down API service.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Production-grade REST API for real-time credit card fraud risk scoring. "
        "Engineered for high-throughput, low-latency transaction inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service metadata."""
    return {
        "service": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Health check endpoint confirming API operational status and model readiness."""
    global predictor_instance
    uptime = time.time() - start_time
    is_loaded = predictor_instance is not None and predictor_instance.model is not None
    model_version = predictor_instance.model_version if is_loaded else "not_loaded"

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=model_version,
        uptime_seconds=round(uptime, 2),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_transaction(
    payload: TransactionInput,
    decision_threshold: float = 0.5,
):
    """
    Score a single credit card transaction for fraud risk.
    """
    t0 = time.perf_counter()
    global predictor_instance
    if predictor_instance is None:
        try:
            predictor_instance = get_predictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifacts not available: {str(e)}. Train model first.",
            )

    try:
        raw_dict = payload.model_dump()
        result = predictor_instance.predict_single(raw_dict, threshold=decision_threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ts = datetime.now(timezone.utc).isoformat()
        response = PredictionResponse(
            fraud_probability=result["fraud_probability"],
            fraud_flag=result["fraud_flag"],
            decision_threshold=result["decision_threshold"],
            model_version=result["model_version"],
            latency_ms=round(latency_ms, 2),
            timestamp=ts,
        )

        # Structured prediction logging
        log_entry = {
            "timestamp": ts,
            "latency_ms": round(latency_ms, 2),
            "amount": raw_dict.get("Amount"),
            "fraud_probability": response.fraud_probability,
            "fraud_flag": response.fraud_flag,
            "decision_threshold": response.decision_threshold,
            "model_version": response.model_version,
        }
        pred_logger.info(json.dumps(log_entry))

        return response
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_batch_transactions(batch_payload: BatchTransactionInput):
    """
    Score multiple credit card transactions in a single batch request.
    """
    t0 = time.perf_counter()
    global predictor_instance
    if predictor_instance is None:
        try:
            predictor_instance = get_predictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifacts not available: {str(e)}",
            )

    try:
        transactions = [t.model_dump() for t in batch_payload.transactions]
        results = predictor_instance.predict_batch(
            transactions, threshold=batch_payload.decision_threshold
        )
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        ts = datetime.now(timezone.utc).isoformat()
        per_item_latency = total_latency_ms / max(len(transactions), 1)

        pred_responses = [
            PredictionResponse(
                fraud_probability=res["fraud_probability"],
                fraud_flag=res["fraud_flag"],
                decision_threshold=res["decision_threshold"],
                model_version=res["model_version"],
                latency_ms=round(per_item_latency, 2),
                timestamp=ts,
            )
            for res in results
        ]

        flagged_count = sum(1 for p in pred_responses if p.fraud_flag)

        return BatchPredictionResponse(
            predictions=pred_responses,
            total_transactions=len(pred_responses),
            flagged_fraud_count=flagged_count,
            total_latency_ms=round(total_latency_ms, 2),
        )
    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference pipeline failed: {str(e)}",
        )
