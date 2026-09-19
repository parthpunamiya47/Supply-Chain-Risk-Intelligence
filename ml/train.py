"""SupplyChain Sentinel - Stage 1: Random Forest training script.

Steps:
  1. Load dataset (data/supply_chain_shipments.csv)
  2. Leak-free train/test split via ml/preprocess.py
  3. Fit preprocessor + RandomForestClassifier (class_weight='balanced'
     because disruptions are the minority class)
  4. Evaluate on the held-out test set (precision/recall/F1/confusion
     matrix/ROC-AUC) - see evaluate.py for the detailed report
  5. Save: models/random_forest_pipeline.joblib (preprocessor+model),
           models/feature_importances.json, models/training_metadata.json

Run:  python ml/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:  # flat import when run as a script, package import when embedded
    from preprocess import TARGET, build_preprocessor, make_train_test_split
except ImportError:  # pragma: no cover
    from ml.preprocess import TARGET, build_preprocessor, make_train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "supply_chain_shipments.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
IMPORTANCE_PATH = MODELS_DIR / "feature_importances.json"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

RANDOM_STATE = 42


def train(df: pd.DataFrame) -> dict:
    """Train the RF pipeline on `df` and persist artifacts. Returns metadata."""
    split = make_train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=True)

    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",  # disruption is the minority class
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    pipeline.fit(split.X_train, split.y_train)

    # ----------------------------- evaluation on the held-out test set -----
    try:
        from evaluate import evaluate_model
    except ImportError:  # pragma: no cover
        from ml.evaluate import evaluate_model

    metrics = evaluate_model(
        pipeline,
        split.X_test,
        split.y_test,
        out_dir=MODELS_DIR,
        prefix="rf",
    )

    # ------------------------------------------- persist model + importance
    joblib.dump(pipeline, MODEL_PATH)

    # Permutation importance on the RAW feature columns (more honest for the
    # demo than pipeline-internal impurity importances, and name-stable).
    from sklearn.inspection import permutation_importance

    print("Computing permutation importances (this can take ~30-60s)...")
    pi = permutation_importance(
        pipeline,
        split.X_test,
        split.y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    order = np.argsort(pi.importances_mean)[::-1]  # descending
    cols = split.X_test.columns.tolist()
    imp_records = [
        {
            "feature": cols[i],
            "importance_mean": round(float(pi.importances_mean[i]), 5),
            "importance_std": round(float(pi.importances_std[i]), 5),
        }
        for i in order
    ]
    IMPORTANCE_PATH.write_text(json.dumps(imp_records, indent=2))

    # ------------------------------------------------- training metadata ---
    meta = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 400,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_rows_total": int(len(df)),
        "n_train": int(len(split.X_train)),
        "n_test": int(len(split.X_test)),
        "features": split.X_test.columns.tolist(),
        "target": TARGET,
        "disruption_rate_train": round(float(split.y_train.mean()), 4),
        "disruption_rate_test": round(float(split.y_test.mean()), 4),
        "test_metrics": metrics,
        "artifacts": {
            "model": MODEL_PATH.name,
            "feature_importances": IMPORTANCE_PATH.name,
            "metadata": METADATA_PATH.name,
        },
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))

    return meta


def main() -> None:
    if not DATA_CSV.exists():
        raise SystemExit(f"Dataset not found at {DATA_CSV}. Run `python ml/generate_dataset.py` first.")
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df):,} rows from {DATA_CSV.name}")

    meta = train(df)

    print("\n--- Training complete ---")
    print(f"Model saved to:  {MODEL_PATH}")
    print(f"Importances:     {IMPORTANCE_PATH}")
    print(f"Metadata:        {METADATA_PATH}")
    print(f"Test set:        {meta['n_test']} rows | disruption rate {meta['disruption_rate_test']:.1%}")
    print(f"ROC-AUC:         {metrics_auc(meta)}")


def metrics_auc(meta: dict) -> str:
    return f"{meta['test_metrics']['roc_auc']:.3f}"


if __name__ == "__main__":
    main()
