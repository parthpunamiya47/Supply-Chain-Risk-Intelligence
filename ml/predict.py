"""SupplyChain Sentinel - Stage 1: Prediction API.

Loads the trained pipeline (preprocessor + RandomForest) from
models/random_forest_pipeline.joblib and exposes:

    predict_risk(input_data: dict) -> dict

returning the Stage-1 contract:

    {
      "risk_probability": 82.4,          # 0-100 scale
      "risk_level": "HIGH",              # LOW (<40) / MEDIUM (40-69) / HIGH (>=70)
      "predicted_disruption": True,      # class 1 at the 0.5 default threshold
      "top_risk_factors": [ "High previous delays", ... ]
    }

Also usable from the CLI:
    python ml/predict.py --input-json '{"supplier_id": "SUP-001", ...}'
    python ml/predict.py --samples      # runs 6 built-in sample shipments

Stage 2 note: the FastAPI layer should import `predict_risk` from this module
(or call this CLI) rather than reimplementing inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:  # flat import when run as a script, package import when embedded in FastAPI
    from explain import (
        build_feature_spreads,
        build_low_risk_baseline,
        explain_prediction,
        load_global_importances,
    )
    from evaluate import risk_level_from_probability
    from preprocess import (
        CATEGORICAL_FEATURES,
        IDENTIFIER_FIELDS,
        NUMERIC_FEATURES,
        TARGET,
    )
except ImportError:  # pragma: no cover - package-style import (Stage 2 backend)
    from ml.explain import (
        build_feature_spreads,
        build_low_risk_baseline,
        explain_prediction,
        load_global_importances,
    )
    from ml.evaluate import risk_level_from_probability
    from ml.preprocess import (
        CATEGORICAL_FEATURES,
        IDENTIFIER_FIELDS,
        NUMERIC_FEATURES,
        TARGET,
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "supply_chain_shipments.csv"
MODEL_PATH = ROOT / "models" / "random_forest_pipeline.joblib"
IMPORTANCE_PATH = ROOT / "models" / "feature_importances.json"

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_path: Path | str | None = None):
    """Load and cache the trained pipeline (preprocessor + classifier)."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first: `python ml/train.py`"
        )
    key = str(path.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = joblib.load(path)
    return _MODEL_CACHE[key]


def _validate_input(input_data: dict[str, Any]) -> dict[str, Any]:
    """Keep only known features; complain about missing required ones.

    Identifier fields (e.g. `supplier_id`) are accepted for traceability but
    stripped - they never reach the model.
    """
    known = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + IDENTIFIER_FIELDS + [TARGET])
    unknown = [k for k in input_data if k not in known]
    if unknown:
        raise ValueError(f"Unknown input fields: {unknown}. Expected features: {NUMERIC_FEATURES}")

    missing = [f for f in NUMERIC_FEATURES if f not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return {k: input_data[k] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES if k in input_data}


def _load_reference_data() -> tuple[pd.Series, pd.Series]:
    """Return (healthy-baseline medians, feature spreads) for explanations.

    Falls back to the given row only when the dataset is unavailable.
    """
    cached = _MODEL_CACHE.get("_reference_data")
    if cached is not None:
        return cached
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        ref = (build_low_risk_baseline(df), build_feature_spreads(df))
    else:
        ref = (pd.Series(dtype=float), pd.Series(dtype=float))
    _MODEL_CACHE["_reference_data"] = ref
    return ref


def _explain_row(row: pd.Series) -> list[str]:
    try:
        importances = load_global_importances()
        baseline, spreads = _load_reference_data()
        if baseline.empty:
            return []
        return explain_prediction(row, baseline, importances, spreads=spreads)["top_risk_factors"]
    except FileNotFoundError:
        return []  # importances not computed yet; factors stay empty


def explain_factors(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Public per-factor contributions for one shipment (used by the Stage 4
    rule-based recommender and any future UX).

    Returns a list of {feature, phrase, value, healthy_median,
    deviation_iqr, contribution} sorted by contribution (descending).
    """
    clean = _validate_input(input_data)
    row = pd.DataFrame([clean], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    try:
        importances = load_global_importances()
        baseline, spreads = _load_reference_data()
        if baseline.empty:
            return []
        return explain_prediction(row.iloc[0], baseline, importances, spreads=spreads)[
            "contributions"
        ]
    except FileNotFoundError:
        return []


def predict_risk(
    input_data: dict[str, Any],
    threshold: float = 0.5,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Predict disruption risk for a single shipment.

    Returns the Stage-1 API contract (see module docstring).
    """
    clean = _validate_input(input_data)
    model = _load_model(model_path)

    row = pd.DataFrame([clean], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    proba = float(model.predict_proba(row)[0, 1])
    proba_pct = round(proba * 100, 1)

    result = {
        "risk_probability": proba_pct,
        "risk_level": risk_level_from_probability(proba_pct),
        "predicted_disruption": bool(proba >= threshold),
        "top_risk_factors": _explain_row(row.iloc[0]),
    }
    return result


def predict_risk_batch(
    rows: list[dict[str, Any]],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Predict for many shipments (single vectorised pass through the model)."""
    if not rows:
        return []
    cleaned = [_validate_input(r) for r in rows]
    model = _load_model()
    frame = pd.DataFrame(cleaned, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    probas = model.predict_proba(frame)[:, 1]

    importances = []
    baseline = spreads = None
    try:
        importances = load_global_importances()
        baseline, spreads = _load_reference_data()
    except FileNotFoundError:
        pass

    out = []
    for clean_row, p in zip(frame.itertuples(index=False), probas):
        pct = round(float(p) * 100, 1)
        res = {
            "risk_probability": pct,
            "risk_level": risk_level_from_probability(pct),
            "predicted_disruption": bool(p >= threshold),
            "top_risk_factors": [],
        }
        if importances and baseline is not None and not baseline.empty:
            res["top_risk_factors"] = explain_prediction(
                pd.Series(clean_row, index=frame.columns), baseline, importances, spreads=spreads
            )["top_risk_factors"]
        out.append(res)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

SAMPLE_SHIPMENTS: list[dict[str, Any]] = [
    {
        "name": "Sample 1 - low-risk lane",
        "supplier_id": "SUP-001",
        "supplier_reliability": 95.0,
        "previous_delays": 0,
        "average_lead_time": 8.0,
        "demand": 1500,
        "inventory_level": 5000,
        "weather_risk": 1.0,
        "transportation_risk": 1.5,
        "distance": 250,
        "supplier_capacity": 40000,
        "shipment_size": 1200,
        "historical_disruptions": 0,
    },
    {
        "name": "Sample 2 - unreliable supplier, many past delays",
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
    },
    {
        "name": "Sample 3 - storm season + risky route",
        "supplier_id": "SUP-012",
        "supplier_reliability": 80.0,
        "previous_delays": 3,
        "average_lead_time": 15.0,
        "demand": 4000,
        "inventory_level": 3000,
        "weather_risk": 9.0,
        "transportation_risk": 8.5,
        "distance": 6200,
        "supplier_capacity": 30000,
        "shipment_size": 3600,
        "historical_disruptions": 1,
    },
    {
        "name": "Sample 4 - razor-thin inventory vs demand",
        "supplier_id": "SUP-020",
        "supplier_reliability": 88.0,
        "previous_delays": 1,
        "average_lead_time": 10.0,
        "demand": 45000,
        "inventory_level": 900,
        "weather_risk": 3.0,
        "transportation_risk": 3.5,
        "distance": 900,
        "supplier_capacity": 90000,
        "shipment_size": 42000,
        "historical_disruptions": 0,
    },
    {
        "name": "Sample 5 - clean record, short lead time",
        "supplier_id": "SUP-031",
        "supplier_reliability": 91.0,
        "previous_delays": 2,
        "average_lead_time": 6.0,
        "demand": 2200,
        "inventory_level": 7000,
        "weather_risk": 2.0,
        "transportation_risk": 2.0,
        "distance": 400,
        "supplier_capacity": 25000,
        "shipment_size": 2000,
        "historical_disruptions": 0,
    },
    {
        "name": "Sample 6 - worst case on every dimension",
        "supplier_id": "SUP-040",
        "supplier_reliability": 48.0,
        "previous_delays": 14,
        "average_lead_time": 35.0,
        "demand": 30000,
        "inventory_level": 4000,
        "weather_risk": 9.5,
        "transportation_risk": 9.0,
        "distance": 9500,
        "supplier_capacity": 10000,
        "shipment_size": 25000,
        "historical_disruptions": 7,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict shipment disruption risk")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-json", type=str, help="single shipment as a JSON string")
    group.add_argument("--input-file", type=str, help="path to a JSON file with shipment fields")
    group.add_argument("--samples", action="store_true", help="run the built-in sample shipments")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.samples or (not args.input_json and not args.input_file):
        payloads = [{k: v for k, v in s.items() if k != "name"} for s in SAMPLE_SHIPMENTS]
        results = predict_risk_batch(payloads, threshold=args.threshold)
        print("=== SupplyChain Sentinel - sample shipment predictions ===\n")
        for sample, res in zip(SAMPLE_SHIPMENTS, results):
            print(f"--- {sample['name']} ---")
            for k, v in res.items():
                if k == "top_risk_factors":
                    print(f"  {k}:")
                    for f in v:
                        print(f"    - {f}")
                else:
                    print(f"  {k}: {v}")
            print()
        return

    payload = json.loads(args.input_json) if args.input_json else json.loads(Path(args.input_file).read_text())
    print(json.dumps(predict_risk(payload, threshold=args.threshold), indent=2))


if __name__ == "__main__":
    main()
