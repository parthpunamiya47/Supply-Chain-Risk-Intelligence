"""SupplyChain Sentinel - Stage 1: Model evaluation.

Reports precision / recall / F1 (per class + macro, since disruption is the
minority class we care about), confusion matrix, ROC-AUC and a
precision-recall trade-off table across decision thresholds.

Writes: models/evaluation_report.txt, models/confusion_matrix.png,
        models/roc_curve.png, models/metrics.json

Run:  python ml/evaluate.py          (uses the saved pipeline + fresh split)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "supply_chain_shipments.csv"
MODELS_DIR = ROOT / "models"
REPORT_PATH = MODELS_DIR / "evaluation_report.txt"

TARGET = "disrupted"


def risk_level_from_probability(p: float) -> str:
    """0-39 LOW, 40-69 MEDIUM, 70-100 HIGH (probability in percent)."""
    if p < 40:
        return "LOW"
    if p < 70:
        return "MEDIUM"
    return "HIGH"


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path = MODELS_DIR,
    prefix: str = "rf",
) -> dict:
    """Compute test metrics, print/save the report. Returns a metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_test)),
        "positive_rate_test": round(float(np.mean(y_test)), 4),
    }

    # ------------------------------------------------------------- report ---
    lines = [
        "=== SupplyChain Sentinel - Model Evaluation (held-out test set) ===",
        f"Test rows: {metrics['n_test']}   |   disruption rate: {metrics['positive_rate_test']:.1%}",
        "",
        f"Precision (disrupted=1): {precision:.4f}",
        f"Recall    (disrupted=1): {recall:.4f}",
        f"F1        (disrupted=1): {f1:.4f}",
        f"Macro F1:                {macro_f1:.4f}",
        f"ROC-AUC:                 {auc:.4f}",
        "",
        "Confusion matrix (rows=actual, cols=predicted):",
        f"            pred 0    pred 1",
        f"actual 0  | {cm[0, 0]:>7} | {cm[0, 1]:>7}",
        f"actual 1  | {cm[1, 0]:>7} | {cm[1, 1]:>7}",
        "",
        "NOTE: disruption events are relatively rare, so precision/recall/F1",
        "and ROC-AUC are the meaningful metrics here - accuracy alone would be",
        "misleading (a model that always predicts 0 scores well on accuracy).",
        "Metrics are on SYNTHETIC data with baked-in relationships; they are not",
        "claims about real-world performance.",
    ]

    # threshold trade-off table (useful when backend wants stricter alerts)
    lines += ["", "Threshold trade-offs:"]
    lines.append(f"{'thr':>5} | {'precision':>9} | {'recall':>9} | {'F1':>7}")
    best = {"thr": 0.5, "f1": 0.0}
    for thr in np.arange(0.2, 0.81, 0.05):
        p = precision_score(y_test, proba >= thr, zero_division=0)
        r = recall_score(y_test, proba >= thr, zero_division=0)
        f = f1_score(y_test, proba >= thr, zero_division=0)
        lines.append(f"{thr:>5.2f} | {p:>9.3f} | {r:>9.3f} | {f:>7.3f}")
        if f > best["f1"]:
            best = {"thr": float(thr), "f1": float(f)}
    lines.append(f"\nBest F1 threshold in scan: {best['thr']:.2f} (F1={best['f1']:.3f})")

    report = "\n".join(lines)
    (out_dir / f"{prefix}_evaluation_report.txt").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(report)

    # ------------------------------------------------------------- plots ----
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["actual 0", "actual 1"])
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close()

    prec, rec, thr = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(4.5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=120)
    plt.close()

    return metrics


def main() -> None:
    """Standalone re-evaluation using the SAVED pipeline + a fresh split."""
    model_path = MODELS_DIR / "random_forest_pipeline.joblib"
    if not model_path.exists():
        raise SystemExit("No saved model found. Run `python ml/train.py` first.")
    if not DATA_CSV.exists():
        raise SystemExit(f"Dataset not found at {DATA_CSV}. Run `python ml/generate_dataset.py` first.")

    import joblib

    try:
        from preprocess import make_train_test_split
    except ImportError:  # pragma: no cover
        from ml.preprocess import make_train_test_split

    model = joblib.load(model_path)
    df = pd.read_csv(DATA_CSV)
    split = make_train_test_split(df, test_size=0.2, random_state=42, stratify=True)
    print(f"Re-evaluating saved model {model_path.name} on a fresh stratified split...\n")
    evaluate_model(model, split.X_test, split.y_test)
    print(f"\nFull report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
