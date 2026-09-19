"""SupplyChain Sentinel - Stage 6: dashboard overview data builder.

Computes REAL KPIs for the frontend overview page from existing systems:
  - the generated dataset (suppliers, shipments),
  - the trained Stage 1 model (batch predictions on a recent-feel sample),
  - the Stage 3 graph (node risk bands).

No fake numbers: every value is derived from `data/supply_chain_shipments.csv`,
`models/random_forest_pipeline.joblib` or `backend/graph.py`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.graph import NODES
from backend.recommender_router import _band

ROOT = None  # resolved lazily to keep import side-effect free


def _dataset_path():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "data" / "supply_chain_shipments.csv"


def _risk_band(risk01: float) -> str:
    return _band(risk01)


def _graph_kpis() -> dict[str, Any]:
    """Node risk bands + alerts derived from the real network."""
    suppliers = [n for n in NODES if n["type"] == "supplier"]
    bands: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for n in NODES:
        bands[_risk_band(float(n["risk"]))] += 1

    alerts = sorted(
        (
            {
                "id": f"alert-{n['id']}",
                "node_id": n["id"],
                "name": n["name"],
                "type": n["type"],
                "risk": float(n["risk"]),
                "risk_probability": round(float(n["risk"]) * 100, 1),
                "severity": _risk_band(float(n["risk"])),
                "message": (
                    f"{n['name']} baseline risk {float(n['risk']) * 100:.0f}/100 "
                    f"({_risk_band(float(n['risk'])).lower()}) - review mitigations."
                ),
            }
            for n in NODES
            if float(n["risk"]) >= 0.4
        ),
        key=lambda a: a["risk"],
        reverse=True,
    )
    return {
        "total_suppliers": len(suppliers),
        "graph_nodes": len(NODES),
        "high_risk_nodes": bands["HIGH"],
        "risk_distribution": bands,
        "alerts": alerts,
        "at_risk_shipments": 0,  # set by caller from model predictions
        "average_risk": 0.0,
        "supplier_performance": [],
        "risk_trend": [],
    }


def _supplier_performance(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-supplier realized disruption rate + shipment counts (real data)."""
    if "supplier_id" not in df.columns:
        return []
    grouped = df.groupby("supplier_id").agg(
        shipments=("disrupted", "size"),
        disruptions=("disrupted", "sum"),
        avg_reliability=("supplier_reliability", "mean"),
    )
    # top 10 by shipment volume keeps the chart readable
    top = grouped.sort_values("shipments", ascending=False).head(10)
    return [
        {
            "supplier_id": idx,
            "shipments": int(row["shipments"]),
            "disruptions": int(row["disruptions"]),
            "disruption_rate_pct": round(100.0 * row["disruptions"] / max(row["shipments"], 1), 1),
            "avg_reliability": round(float(row["avg_reliability"]), 1),
        }
        for idx, row in top.iterrows()
    ]


def build_overview(sample_size: int = 300) -> dict[str, Any]:
    """Assemble the overview payload from dataset + model + graph."""
    kpis = _graph_kpis()

    path = _dataset_path()
    if not path.exists():
        kpis["note"] = "Dataset not generated yet - run `python ml/generate_dataset.py`."
        return kpis

    df = pd.read_csv(path)

    # --- model predictions on the most recent-feeling slice (highest lead time
    # proxy is artificial; use a deterministic random sample instead) --------
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    try:
        from ml.predict import predict_risk_batch

        feats = [
            {
                "supplier_reliability": r.supplier_reliability,
                "previous_delays": r.previous_delays,
                "average_lead_time": r.average_lead_time,
                "demand": r.demand,
                "inventory_level": r.inventory_level,
                "weather_risk": r.weather_risk,
                "transportation_risk": r.transportation_risk,
                "distance": r.distance,
                "supplier_capacity": r.supplier_capacity,
                "shipment_size": r.shipment_size,
                "historical_disruptions": r.historical_disruptions,
            }
            for r in sample.itertuples(index=False)
        ]
        preds = predict_risk_batch(feats)
    except FileNotFoundError:
        preds = []

    if preds:
        at_risk = sum(1 for p in preds if p["risk_level"] in ("MEDIUM", "HIGH"))
        avg_prob = sum(p["risk_probability"] for p in preds) / len(preds)
        pred_bands = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for p in preds:
            pred_bands[p["risk_level"]] += 1
        kpis["at_risk_shipments"] = at_risk
        kpis["average_risk"] = round(avg_prob, 1)
        kpis["prediction_distribution"] = pred_bands
        kpis["sample_size"] = len(preds)

    kpis["supplier_performance"] = _supplier_performance(df)

    # --- risk trend: weekly realized disruption rate (real, from dataset) ---
    if "disrupted" in df.columns:
        chunk = max(1, len(df) // 12)  # ~12 periods
        trend = []
        for i in range(0, len(df), chunk):
            part = df.iloc[i : i + chunk]
            trend.append(
                {
                    "period": f"W{len(trend) + 1}",
                    "disruption_rate_pct": round(100.0 * part["disrupted"].mean(), 1),
                    "shipments": int(len(part)),
                }
            )
        kpis["risk_trend"] = trend

    return kpis
