"""SupplyChain Sentinel - Stage 1: Reusable preprocessing pipeline.

Builds a scikit-learn ColumnTransformer that:
  - imputes missing numeric values (median, fitted on TRAIN only -> no leakage)
  - passes numeric features through (trees need no scaling; scaler optional)
  - is generic enough to also handle categorical features if the schema grows
    (use `include_categorical=True`; handled via one-hot encoding)

Also owns:
  - the canonical feature list (single source of truth for training & serving)
  - a leak-free train/test split helper

Every later stage (FastAPI backend, what-if simulator) should import
`build_preprocessor` / `load_feature_names` from here so train & serve agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- #
# Canonical schema - single source of truth
# --------------------------------------------------------------------------- #

NUMERIC_FEATURES = [
    "supplier_reliability",
    "previous_delays",
    "average_lead_time",
    "demand",
    "inventory_level",
    "weather_risk",
    "transportation_risk",
    "distance",
    "supplier_capacity",
    "shipment_size",
    "historical_disruptions",
]

# Dataset currently ships `supplier_id` as an identifier; it is stored but NOT
# used as a model feature (IDs don't generalise). If real categorical features
# (e.g. transport_mode, region) are added later, list them here.
CATEGORICAL_FEATURES: list[str] = []

# Identifier columns accepted (and ignored) by the prediction API for
# traceability - they never reach the model.
IDENTIFIER_FIELDS: list[str] = ["supplier_id"]

TARGET = "disrupted"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_feature_names() -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features)."""
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)


def build_preprocessor(include_categorical: bool = False, scale: bool = False) -> ColumnTransformer:
    """Build the fitted-at-train-time preprocessing transformer.

    - Median imputation for numerics (robust to skewed counts/distances).
    - Optional StandardScaler (trees don't need it; handy if a linear model is
      swapped in later).
    - Optional one-hot encoding for categorical features.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()) if scale else ("passthrough", "passthrough"),
        ]
    )

    transformers: list[tuple[str, object, list[str]]] = [
        ("num", num_pipe, list(NUMERIC_FEATURES)),
    ]

    if include_categorical and CATEGORICAL_FEATURES:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, list(CATEGORICAL_FEATURES)))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Split:
    """Leak-free split: the test set is carved out BEFORE any fitting."""
    from sklearn.model_selection import train_test_split

    X = df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    return Split(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# imported late to keep the dataclass section readable
from sklearn.preprocessing import StandardScaler  # noqa: E402
