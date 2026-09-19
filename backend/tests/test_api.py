"""SupplyChain Sentinel - Stage 2: API tests (pytest).

Covers Task 6:
  - GET  /api/health
  - POST /api/predict-risk  (valid -> contract shape, ranges, risk levels)
  - POST /api/predict-risk  (invalid inputs -> 422 with field details)
  - POST /api/predict-risk/batch
  - CORS preflight (Task 5)

Run from the project root:  .venv/Scripts/python -m pytest backend/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # lifespan runs: loads + warms the trained Stage 1 model
    with TestClient(app) as c:
        yield c


VALID_SHIPMENT: dict = {
    "supplier_id": "SUP-007",
    "supplier_reliability": 52.0,
    "previous_delays": 12,
    "average_lead_time": 22.0,
    "demand": 9000,
    "inventory_level": 2500,
    "weather_risk": 5.5,
    "transportation_risk": 6.0,
    "distance": 3800,
    "supplier_capacity": 15000,
    "shipment_size": 8000,
    "historical_disruptions": 5,
}


# ------------------------------------------------------------- health ----
def test_health_returns_healthy(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ------------------------------------------------------ valid prediction ----
def test_predict_valid_shipment_matches_contract(client):
    res = client.post("/api/predict-risk", json=VALID_SHIPMENT)
    assert res.status_code == 200
    data = res.json()

    # exact Task 2 contract
    assert set(data.keys()) == {"risk_probability", "risk_level", "predicted_disruption", "top_risk_factors"}
    assert isinstance(data["risk_probability"], (int, float))
    assert 0 <= data["risk_probability"] <= 100
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(data["predicted_disruption"], bool)
    assert isinstance(data["top_risk_factors"], list)

    # risk-level banding agrees with the probability
    p = data["risk_probability"]
    expected = "LOW" if p < 40 else "MEDIUM" if p < 70 else "HIGH"
    assert data["risk_level"] == expected
    # class agrees with the 0.5 threshold
    assert data["predicted_disruption"] == (p / 100 >= 0.5)


def test_predict_high_risk_shipment_is_high(client):
    worst = dict(VALID_SHIPMENT)
    worst.update(
        supplier_reliability=48.0,
        previous_delays=14,
        average_lead_time=35.0,
        demand=30000,
        inventory_level=4000,
        weather_risk=9.5,
        transportation_risk=9.0,
        distance=9500,
        supplier_capacity=10000,
        shipment_size=25000,
        historical_disruptions=7,
    )
    res = client.post("/api/predict-risk", json=worst)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == "HIGH"
    assert data["predicted_disruption"] is True
    assert len(data["top_risk_factors"]) > 0


def test_predict_low_risk_shipment_is_low(client):
    best = dict(VALID_SHIPMENT)
    best.update(
        supplier_id="SUP-001",
        supplier_reliability=95.0,
        previous_delays=0,
        average_lead_time=8.0,
        demand=1500,
        inventory_level=5000,
        weather_risk=1.0,
        transportation_risk=1.5,
        distance=250,
        supplier_capacity=40000,
        shipment_size=1200,
        historical_disruptions=0,
    )
    res = client.post("/api/predict-risk", json=best)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == "LOW"
    assert data["predicted_disruption"] is False


# ------------------------------------------------------ invalid inputs ----
def test_missing_required_field_returns_422(client):
    body = {k: v for k, v in VALID_SHIPMENT.items() if k != "previous_delays"}
    res = client.post("/api/predict-risk", json=body)
    assert res.status_code == 422
    fields = {e["loc"][-1] for e in res.json()["detail"]}
    assert "previous_delays" in fields


def test_out_of_range_values_return_422(client):
    cases = [
        {"supplier_reliability": 150.0},   # > 100
        {"weather_risk": -1.0},            # < 0
        {"transportation_risk": 11.0},     # > 10
        {"average_lead_time": 0.0},        # must be > 0
        {"distance": -100.0},              # negative
    ]
    for override in cases:
        body = {**VALID_SHIPMENT, **override}
        res = client.post("/api/predict-risk", json=body)
        assert res.status_code == 422, f"{override} should be rejected"
        field = next(iter(override))
        assert any(e["loc"][-1] == field for e in res.json()["detail"]), f"error must name {field}"


def test_wrong_type_returns_422(client):
    res = client.post("/api/predict-risk", json={**VALID_SHIPMENT, "demand": "lots"})
    assert res.status_code == 422
    assert any(e["loc"][-1] == "demand" for e in res.json()["detail"])


def test_invalid_threshold_returns_422(client):
    res = client.post("/api/predict-risk?threshold=1.5", json=VALID_SHIPMENT)
    assert res.status_code == 422


# ------------------------------------------------------------- batch ----
def test_batch_predictions(client):
    low = dict(VALID_SHIPMENT, supplier_reliability=95.0, previous_delays=0, weather_risk=1.0)
    high = dict(VALID_SHIPMENT, supplier_reliability=48.0, previous_delays=14, weather_risk=9.5)
    res = client.post("/api/predict-risk/batch", json=[low, high])
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list) and len(data) == 2
    for item in data:
        assert set(item.keys()) == {
            "risk_probability", "risk_level", "predicted_disruption", "top_risk_factors"
        }
    assert data[0]["risk_probability"] < data[1]["risk_probability"]


def test_empty_batch_returns_422(client):
    res = client.post("/api/predict-risk/batch", json=[])
    assert res.status_code == 422


# ------------------------------------------------------------- CORS ----
def test_cors_preflight_allowed_origin(client):
    res = client.options(
        "/api/predict-risk",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_headers_on_actual_request(client):
    res = client.post("/api/predict-risk", json=VALID_SHIPMENT, headers={"Origin": "http://localhost:5173"})
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    res = client.options(
        "/api/predict-risk",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.status_code == 400
    assert "access-control-allow-origin" not in res.headers
