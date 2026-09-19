"""SupplyChain Sentinel - Stage 1: Synthetic dataset generator.

Creates ~N shipment records (default 2,500) with realistic supply-chain
features and a `disrupted` target whose probability depends on the features
in a meaningful, interpretable way (NOT random noise).

Feature -> risk relationship baked in:
  - low supplier_reliability        -> higher disruption risk
  - high previous_delays            -> higher risk
  - long average_lead_time          -> higher risk (longer exposure window)
  - high weather_risk               -> higher risk
  - high transportation_risk        -> higher risk
  - high historical_disruptions     -> higher risk
  - low inventory_level (vs demand) -> higher risk (business impact)
  - high demand                     -> higher risk
  - distance / capacity / shipment_size -> mild secondary effects

Run:  python ml/generate_dataset.py            (writes data/supply_chain_shipments.csv)
      python ml/generate_dataset.py --rows 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "supply_chain_shipments.csv"

RANDOM_SEED = 42

N_SUPPLIERS = 40  # suppliers repeat -> realistic supplier-level patterns

NUMERIC_FEATURES = [
    "supplier_reliability",   # 0-100 score (higher = more reliable)
    "previous_delays",        # count of delays in past 6 months
    "average_lead_time",      # days
    "demand",                 # units demanded at destination
    "inventory_level",        # units in stock at destination
    "weather_risk",           # 0-10
    "transportation_risk",    # 0-10
    "distance",               # km
    "supplier_capacity",      # units/month
    "shipment_size",          # units
    "historical_disruptions", # disruptions tied to this supplier/route in past year
]

FEATURES = NUMERIC_FEATURES  # (schema keeps room for categorical columns later)

TARGET = "disrupted"


def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_shipments(n_rows: int = 2500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic shipment dataset and return it as a DataFrame."""
    rng = np.random.default_rng(seed)
    n_rows = int(n_rows)

    # --- supplier-level base traits (suppliers have persistent behaviour) ----
    supplier_ids = np.array([f"SUP-{i:03d}" for i in range(1, N_SUPPLIERS + 1)])
    supplier_reliability = np.round(rng.uniform(45, 98, N_SUPPLIERS), 1)
    supplier_prev_delays = rng.integers(0, 15, N_SUPPLIERS)
    supplier_hist_disruptions = rng.integers(0, 8, N_SUPPLIERS)

    rows = []
    for _ in range(n_rows):
        s = rng.integers(0, N_SUPPLIERS)

        reliability = float(supplier_reliability[s])
        prev_delays = int(supplier_prev_delays[s])
        hist_disruptions = int(supplier_hist_disruptions[s])

        # shipment-level traits, correlated with supplier traits where sensible
        lead_time = float(np.clip(rng.normal(12 + (100 - reliability) * 0.12, 4), 2, 45))
        weather_risk = float(np.clip(rng.beta(2, 5) * 10, 0, 10))          # mostly low, tail high
        transport_risk = float(np.clip(rng.beta(2.5, 4.5) * 10, 0, 10))
        distance = float(np.clip(rng.lognormal(6.3, 0.75), 50, 12000))     # km, right-skewed
        demand = int(np.clip(rng.lognormal(8.0, 0.8), 100, 60000))
        # inventory is often tight relative to demand -> creates pressure
        inventory = int(np.clip(demand * rng.uniform(0.2, 3.5), 0, 250000))
        capacity = int(np.clip(demand * rng.uniform(0.8, 6.0), 500, 500000))
        shipment_size = int(np.clip(demand * rng.uniform(0.5, 1.4), 10, 120000))

        rows.append(
            {
                "supplier_id": supplier_ids[s],
                "supplier_reliability": reliability,
                "previous_delays": prev_delays,
                "average_lead_time": round(lead_time, 1),
                "demand": demand,
                "inventory_level": inventory,
                "weather_risk": round(weather_risk, 1),
                "transportation_risk": round(transport_risk, 1),
                "distance": round(distance),
                "supplier_capacity": capacity,
                "shipment_size": shipment_size,
                "historical_disruptions": hist_disruptions,
            }
        )

    df = pd.DataFrame(rows)

    # --- inject a few realistic data-quality quirks --------------------------
    # ~1.5% missing weather_risk (station outage), ~1% missing distance
    n_missing_weather = int(0.015 * n_rows)
    n_missing_distance = int(0.01 * n_rows)
    df.loc[rng.choice(n_rows, n_missing_weather, replace=False), "weather_risk"] = np.nan
    df.loc[rng.choice(n_rows, n_missing_distance, replace=False), "distance"] = np.nan

    # ~0.5% duplicated rows (double-entered shipments)
    n_dupes = max(1, int(0.005 * n_rows))
    df = pd.concat([df, df.sample(n_dupes, random_state=seed)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    # --- label generation: risk score -> logistic probability ----------------
    # Components are in roughly comparable units so coefficients stay readable.
    inv_ratio = df["inventory_level"] / df["demand"].clip(lower=1)  # coverage of demand

    risk_score = (
        0.55 * (60.0 - df["supplier_reliability"]) / 10.0        # low reliability => +
        + 0.45 * df["previous_delays"] / 2.0                     # past delays     => +
        + 0.35 * df["historical_disruptions"]                    # supplier record => +
        + 0.40 * df["weather_risk"].fillna(df["weather_risk"].median())  # weather => +
        + 0.50 * df["transportation_risk"]                       # transport       => +
        + 0.20 * (df["average_lead_time"] - 12.0) / 3.0          # long exposure   => +
        + 0.30 * np.log1p(df["distance"].fillna(df["distance"].median())) / 2.5
        - 0.55 * np.log1p(inv_ratio) / 0.35                      # low inventory   => +
        + 0.15 * (df["demand"] - 3000.0) / 3000.0                # high demand     => +
        - 0.10 * np.log1p(df["supplier_capacity"]) / 2.0         # big capacity    => -
        + 0.10 * np.log1p(df["shipment_size"]) / 2.0             # big shipments   => +
    )

    # per-shipment randomness (things no feature captures)
    noise = rng.normal(0, 0.8, len(df))

    # Rescale so per-feature effects stay meaningful, then calibrate the
    # intercept numerically so the OVERALL disruption rate hits the target
    # (~27%) - disruptions must be the *minority* class.
    raw_score = risk_score.values * 0.85 + noise
    target_rate = 0.27
    lo, hi = -12.0, 12.0
    for _ in range(80):  # bisection on the intercept
        mid = (lo + hi) / 2
        if _logistic(raw_score + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    p_disrupt = _logistic(raw_score + intercept)
    df[TARGET] = (rng.random(len(df)) < p_disrupt).astype(int)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic supply-chain shipment data")
    parser.add_argument("--rows", type=int, default=2500, help="number of shipment records")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=str, default=str(DEFAULT_CSV))
    args = parser.parse_args()

    df = generate_shipments(n_rows=args.rows, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    rate = df[TARGET].mean() * 100
    print(f"Wrote {len(df)} shipment records -> {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Disruption rate: {rate:.1f}%  ({int(df[TARGET].sum())} disrupted / {len(df)} total)")


if __name__ == "__main__":
    main()
