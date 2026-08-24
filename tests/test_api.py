"""
Integration and schema tests for FastAPI serving endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import TransactionInput


@pytest.fixture
def client():
    """Create FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_payload():
    """Valid transaction sample payload."""
    payload = {
        "Time": 406.0,
        "Amount": 149.62,
    }
    for i in range(1, 29):
        payload[f"V{i}"] = 0.05 * i
    return payload


def test_root_endpoint(client):
    """Test GET / endpoint metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint(client):
    """Test GET /health operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_seconds" in data


def test_predict_validation_error_negative_amount(client, sample_payload):
    """Test validation error when Amount is negative."""
    invalid_payload = dict(sample_payload)
    invalid_payload["Amount"] = -50.0  # ge=0.0 constraint violation

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_validation_error_missing_field(client, sample_payload):
    """Test validation error when a required column (e.g. V14) is missing."""
    invalid_payload = dict(sample_payload)
    del invalid_payload["V14"]

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_mock_or_real(client, sample_payload):
    """
    Test POST /predict if model artifacts exist or returns 503 with informative message.
    """
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert "fraud_probability" in data
    assert "fraud_flag" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert isinstance(data["fraud_flag"], bool)
    assert "latency_ms" in data
    assert "timestamp" in data


def test_batch_prediction_endpoint(client, sample_payload):
    """Test POST /predict/batch endpoint."""
    batch_payload = {
        "transactions": [sample_payload, sample_payload],
        "decision_threshold": 0.5,
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["total_transactions"] == 2
    assert "flagged_fraud_count" in data
    assert len(data["predictions"]) == 2
    assert "total_latency_ms" in data
