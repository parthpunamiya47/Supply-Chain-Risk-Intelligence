"""SupplyChain Sentinel - Stage 1: Exploratory Data Analysis.

Loads the generated dataset, prints a health & quality report and saves plots:
  - shape, dtypes, missing values, duplicate rows
  - numeric feature distributions (histograms)
  - class balance of the target (`disrupted`)
  - correlation heatmap (Pearson, features + target)
  - feature-wise disruption-rate table (sanity check that signal exists)

Run:  python ml/analyze_data.py
Outputs: console report + PNGs + CSVs in data/analysis/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_CSV = DATA_DIR / "supply_chain_shipments.csv"
ANALYSIS_DIR = DATA_DIR / "analysis"

TARGET = "disrupted"


def h(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def analyze(df: pd.DataFrame) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- overview
    h("1. OVERVIEW")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes.to_string())

    # ------------------------------------------------------- missing values
    h("2. MISSING VALUES")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        print("No missing values.")
    else:
        miss_df = pd.DataFrame({"missing_count": miss, "missing_pct": (miss / len(df) * 100).round(2)})
        print(miss_df.to_string())
        miss_df.to_csv(ANALYSIS_DIR / "missing_values.csv")

    # ---------------------------------------------------------- duplicates
    h("3. DUPLICATES")
    n_dupes = int(df.duplicated().sum())
    print(f"Duplicate rows: {n_dupes} ({n_dupes / len(df) * 100:.2f}%)")
    if n_dupes:
        print(df[df.duplicated(keep=False)].head(10).to_string())

    # ------------------------------------------------------- distributions
    h("4. FEATURE DISTRIBUTIONS")
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET)
    print(df[num_cols].describe().T.round(2).to_string())

    df[num_cols].hist(figsize=(14, 10), bins=30, edgecolor="black")
    plt.suptitle("Feature distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot -> {ANALYSIS_DIR / 'distributions.png'}")

    # ----------------------------------------------------- class imbalance
    h("5. CLASS IMBALANCE (target = 'disrupted')")
    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
    for cls in counts.index:
        print(f"  class {cls}: {counts[cls]:,} rows ({pct[cls]:.1f}%)")
    imb = pct.get(1, 0) / max(pct.get(0, 1e-9), 1e-9)
    print(f"  minority:majority ratio (1:0) = 1 : {1 / imb:.1f}" if imb > 0 else "")
    counts.plot(kind="bar", color=["#4c72b0", "#c44e52"])
    plt.title("Target class balance")
    plt.xlabel("disrupted")
    plt.ylabel("count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "class_balance.png", dpi=120)
    plt.close()
    print(f"Saved plot -> {ANALYSIS_DIR / 'class_balance.png'}")

    # --------------------------------------------------------- correlations
    h("6. CORRELATIONS (Pearson)")
    corr = df[list(num_cols) + [TARGET]].corr()
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap (features vs target)")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()
    print(f"\nSaved plot -> {ANALYSIS_DIR / 'correlation_heatmap.png'}")

    # ------------------------------------------- signal sanity-check table
    h("7. DISRUPTION RATE BY FEATURE QUARTILE (signal check)")
    rows = []
    for col in num_cols:
        try:
            buckets = pd.qcut(df[col], q=4, duplicates="drop")
        except ValueError:
            continue
        rate = df.groupby(buckets, observed=True)[TARGET].mean() * 100
        for interval, r in rate.items():
            rows.append({"feature": col, "bucket": str(interval), "disruption_rate_%": round(r, 1)})
    signal_df = pd.DataFrame(rows)
    print(signal_df.to_string(index=False))
    signal_df.to_csv(ANALYSIS_DIR / "disruption_rate_by_feature.csv", index=False)

    print(
        "\nNOTE: rates are from synthetic generation rules; they demonstrate the baked-in\n"
        "signal, not real-world accuracy claims."
    )


def main() -> None:
    if not DATASET_CSV.exists():
        raise SystemExit(f"Dataset not found at {DATASET_CSV}. Run `python ml/generate_dataset.py` first.")
    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded {DATASET_CSV.name} ({len(df):,} rows)")
    analyze(df)


if __name__ == "__main__":
    main()
