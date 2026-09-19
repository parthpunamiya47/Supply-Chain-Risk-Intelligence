"""SupplyChain Sentinel - Stage 6: overview endpoint tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402
from backend.graph import NODES  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_overview_contract(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    for key in ("total_suppliers", "high_risk_nodes", "risk_distribution",
                "alerts", "at_risk_shipments", "average_risk",
                "supplier_performance", "risk_trend"):
        assert key in data


def test_overview_values_derived_from_real_data(client):
    data = client.get("/api/overview").json()
    assert data["total_suppliers"] == sum(1 for n in NODES if n["type"] == "supplier")
    assert data["graph_nodes"] == len(NODES)
    assert sum(data["risk_distribution"].values()) == len(NODES)
    # alerts only contain genuinely risky nodes (>= 0.4 baseline)
    assert all(a["risk"] >= 0.4 for a in data["alerts"])
    # supplier performance comes from the dataset
    if data["supplier_performance"]:
        sp = data["supplier_performance"][0]
        assert sp["shipments"] > 0
        assert 0 <= sp["disruption_rate_pct"] <= 100
    # trend has ~12 periods of real disruption rates
    assert 0 < len(data["risk_trend"]) <= 13
    # model-based KPIs present when the model artifact exists
    if data.get("sample_size"):
        assert data["at_risk_shipments"] <= data["sample_size"]
        assert 0 <= data["average_risk"] <= 100
        assert sum(data["prediction_distribution"].values()) == data["sample_size"]
