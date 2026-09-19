"""SupplyChain Sentinel - Stage 1: Explainability.

Two complementary views of WHY the model predicts what it predicts:
  1. Global: permutation importances (models/feature_importances.json,
     produced by train.py) -> "which features matter most overall"
  2. Local: for a single shipment, compare its risky features against the
     dataset's healthy/low-risk baseline -> "top_risk_factors"

`top_risk_factors` maps the model's numeric drivers to plain-English reasons
like "High previous delays" or "Low supplier reliability" - exactly what the
Stage 2 FastAPI layer will surface to the frontend.

Run:  python ml/explain.py   (prints global importances + a sample explanation)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "supply_chain_shipments.csv"
MODELS_DIR = ROOT / "models"
IMPORTANCE_PATH = MODELS_DIR / "feature_importances.json"

# --------------------------------------------------------------------------- #
# Global importance helpers
# --------------------------------------------------------------------------- #


def load_global_importances() -> list[dict]:
    """Load permutation importances produced by train.py."""
    if not IMPORTANCE_PATH.exists():
        raise SystemExit("feature_importances.json not found - run `python ml/train.py` first.")
    return json.loads(IMPORTANCE_PATH.read_text())


def influence_label(mean_importance: float) -> str:
    """Map an importance value to a qualitative influence bucket."""
    if mean_importance >= 0.03:
        return "High influence"
    if mean_importance >= 0.01:
        return "Medium influence"
    if mean_importance > 0.0:
        return "Low influence"
    return "No/negative influence (not used by model)"


def print_global_importances() -> None:
    imps = load_global_importances()
    print("=== Global feature importance (permutation, scored on ROC-AUC) ===")
    for rec in imps:
        print(f"  {rec['feature']:<26} {rec['importance_mean']:>8.4f}  ({influence_label(rec['importance_mean'])})")


# --------------------------------------------------------------------------- #
# Local (per-shipment) explanation
# --------------------------------------------------------------------------- #

# risk direction: does a HIGH value of the feature push disruption risk UP?
HIGH_IS_RISKY = {
    "supplier_reliability": False,  # LOW reliability is risky
    "previous_delays": True,
    "average_lead_time": True,
    "demand": True,
    "inventory_level": False,       # LOW inventory is risky
    "weather_risk": True,
    "transportation_risk": True,
    "distance": True,
    "supplier_capacity": False,
    "shipment_size": True,
    "historical_disruptions": True,
}

# Plain-English templates per feature (used for top_risk_factors)
FACTOR_PHRASES = {
    "supplier_reliability": "Low supplier reliability",
    "previous_delays": "High previous delays",
    "average_lead_time": "Long average lead time",
    "demand": "High demand",
    "inventory_level": "Low inventory level",
    "weather_risk": "High weather risk",
    "transportation_risk": "High transportation risk",
    "distance": "Long shipping distance",
    "supplier_capacity": "Tight supplier capacity",
    "shipment_size": "Large shipment size",
    "historical_disruptions": "Past disruptions on this lane",
}


def build_low_risk_baseline(df: pd.DataFrame) -> pd.Series:
    """Median profile of shipments that completed normally (disrupted=0)."""
    healthy = df[df["disrupted"] == 0]
    return healthy.median(numeric_only=True)


def build_feature_spreads(df: pd.DataFrame) -> pd.Series:
    """Per-feature spread (IQR) used to normalise deviations across features
    with very different scales (demand ~1e3-1e5 vs weather_risk 0-10)."""
    num_cols = [c for c in df.columns if c in HIGH_IS_RISKY]
    q1 = df[num_cols].quantile(0.25)
    q3 = df[num_cols].quantile(0.75)
    spread = (q3 - q1).replace(0, np.nan).fillna(1.0)
    return spread


def compute_risk_factors(
    row: pd.Series,
    baseline: pd.Series,
    importances: list[dict],
    top_k: int = 3,
) -> list[str]:
    """Return plain-English `top_risk_factors` for one shipment row.

    Each feature gets a signed deviation from the healthy baseline (positive =
    on the risky side), weighted by its global importance. The top-k risky
    features become human-readable reasons.
    """
    return explain_prediction(row, baseline, importances, top_k=top_k)["top_risk_factors"]


def explain_prediction(
    row: pd.Series,
    baseline: pd.Series,
    importances: list[dict],
    top_k: int = 3,
    spreads: pd.Series | None = None,
) -> dict:
    """Full local explanation: top factors + per-feature contributions.

    A feature qualifies only if it deviates meaningfully from the healthy
    baseline (>= DEADBAND IQRs in the risky direction) - so "normal-looking"
    values are never reported as risk factors. Ranking score = capped
    normalised deviation x global permutation importance.
    """
    DEADBAND = 0.5   # IQRs of risky-side deviation before a factor counts
    Z_CAP = 3.0      # cap extreme z-scores so one huge feature can't dominate
    if spreads is None:
        spreads = pd.Series(1.0, index=baseline.index)
    scored = []
    imp_map = {r["feature"]: max(r["importance_mean"], 0.0) for r in importances}
    for feat, risky_high in HIGH_IS_RISKY.items():
        if feat not in row.index or feat not in baseline.index:
            continue
        val, base = row[feat], baseline[feat]
        if pd.isna(val):
            continue
        signed_risk = (float(val) - float(base)) if risky_high else (float(base) - float(val))
        z = signed_risk / float(spreads.get(feat, 1.0))
        if z < DEADBAND:
            continue  # within normal variation -> not a risk factor
        z_capped = min(z, Z_CAP)
        scored.append(
            {
                "feature": feat,
                "value": float(val),
                "healthy_median": float(base),
                "deviation_iqr": round(float(z), 2),
                "contribution": round(z_capped * imp_map.get(feat, 0.0), 4),
                "phrase": FACTOR_PHRASES[feat],
            }
        )
    scored.sort(key=lambda d: d["contribution"], reverse=True)
    top = [d["phrase"] for d in scored[:top_k]]
    return {"top_risk_factors": top, "contributions": scored}


def main() -> None:
    if not DATA_CSV.exists():
        raise SystemExit(f"Dataset not found at {DATA_CSV}. Run `python ml/generate_dataset.py` first.")

    print_global_importances()

    df = pd.read_csv(DATA_CSV)
    baseline = build_low_risk_baseline(df)
    spreads = build_feature_spreads(df)
    importances = load_global_importances()

    # demo: explain a high-risk-looking shipment from the dataset
    demo = df.loc[(df["previous_delays"] >= 10) & (df["weather_risk"] >= 7)]
    row = (demo.iloc[0] if len(demo) else df.sort_values("previous_delays", ascending=False).iloc[0])
    print("\n=== Example local explanation ===")
    print(row.drop(labels=["disrupted"]).to_string())
    explanation = explain_prediction(row, baseline, importances, spreads=spreads)
    print("\ntop_risk_factors:")
    for r in explanation["top_risk_factors"]:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
