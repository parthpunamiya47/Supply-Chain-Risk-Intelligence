"""SupplyChain Sentinel - Stage 1: end-to-end pipeline runner.

Runs the full Stage 1 flow in order:
  1. generate synthetic dataset  -> data/supply_chain_shipments.csv
  2. exploratory data analysis   -> console + data/analysis/*.png
  3. train + evaluate RF model   -> models/*
  4. explainability summary      -> models/feature_importances.json
  5. demo predictions            -> 6 sample shipments

Run:  python ml/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_data
import explain
import generate_dataset
import train


def main() -> None:
    print("=" * 70)
    print("SupplyChain Sentinel - Stage 1 pipeline")
    print("=" * 70)

    print("\n[1/5] Generating synthetic dataset...")
    generate_dataset.main()

    print("\n[2/5] Running exploratory data analysis...")
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "supply_chain_shipments.csv")
    analyze_data.analyze(df)

    print("\n[3/5] Training Random Forest + evaluating...")
    meta = train.train(df)
    print(f"\nHeld-out test ROC-AUC: {meta['test_metrics']['roc_auc']:.3f} | "
          f"precision: {meta['test_metrics']['precision']:.3f} | "
          f"recall: {meta['test_metrics']['recall']:.3f}")

    print("\n[4/5] Explainability summary...")
    explain.print_global_importances()

    print("\n[5/5] Demo predictions on sample shipments...")
    from predict import SAMPLE_SHIPMENTS, predict_risk_batch

    payloads = [{k: v for k, v in s.items() if k != "name"} for s in SAMPLE_SHIPMENTS]
    results = predict_risk_batch(payloads)
    for sample, res in zip(SAMPLE_SHIPMENTS, results):
        factors = "; ".join(res["top_risk_factors"]) or "-"
        print(
            f"  {sample['name']:<48} "
            f"p={res['risk_probability']:>5.1f}%  "
            f"{res['risk_level']:<6}  "
            f"disruption={res['predicted_disruption']}  "
            f"factors=[{factors}]"
        )

    print("\nDone. Artifacts: data/supply_chain_shipments.csv, data/analysis/*, models/*")


if __name__ == "__main__":
    main()
