"""SupplyChain Sentinel - Stage 2: FastAPI backend.

Endpoints:
  GET  /api/health             -> {"status": "healthy", ...}
  POST /api/predict-risk       -> Stage 1 prediction contract
  POST /api/predict-risk/batch -> list of predictions (single model pass)

Start locally (from the project root):
  Windows (Git Bash):  .venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000
  macOS/Linux:         .venv/bin/python -m uvicorn backend.main:app --reload --port 8000

Then:
  - Swagger UI:  http://127.0.0.1:8000/docs
  - Health:      curl http://127.0.0.1:8000/api/health

The service loads models/random_forest_pipeline.joblib once at startup and
reuses it for every request. Inference is delegated to Stage 1's
`ml.predict.predict_risk` so the API can never drift from the trained model.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `ml.` imports regardless of uvicorn cwd

from ml.predict import MODEL_PATH, predict_risk, predict_risk_batch

from backend.graph_router import router as graph_router
from backend.overview_router import router as overview_router
from backend.recommender_router import router as recommender_router
from backend.simulation_router import router as simulation_router
from backend.schemas import HealthResponse, RiskResponse, ShipmentFeatures

MODEL_VERSION = "stage1-rf-v1"
MAX_BATCH_SIZE = 500

# Warm-up payload used to verify the model end-to-end at startup.
_WARMUP_PAYLOAD: dict[str, float | int] = {
    "supplier_reliability": 80.0,
    "previous_delays": 2,
    "average_lead_time": 10.0,
    "demand": 3000,
    "inventory_level": 5000,
    "weather_risk": 2.0,
    "transportation_risk": 2.0,
    "distance": 500.0,
    "supplier_capacity": 20000,
    "shipment_size": 2500,
    "historical_disruptions": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load + warm the model once at startup; fail fast if the artifact is missing."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python ml/train.py` first."
        )
    predict_risk(_WARMUP_PAYLOAD)  # loads model, reference data, importances into cache
    app.state.model_loaded = True
    yield
    app.state.model_loaded = False


app = FastAPI(
    title="SupplyChain Sentinel API",
    description=(
        "Shipment disruption-risk predictions powered by the Stage 1 "
        "Random Forest model. The model is trained on synthetic data, so "
        "outputs demonstrate the pipeline - they are not real-world risk scores."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# ------------------------------------------------------------ CORS (Task 5) --
# The React frontend will run on a dev origin (Vite/CRA) and call this API.
# Origins are configurable via CORS_ORIGINS (comma-separated) for deployment.
_origins_env = os.environ.get("CORS_ORIGINS", "")
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Stage 3: supply-chain graph + risk propagation
app.include_router(graph_router)

# Stage 4: rule-based mitigation recommendations
app.include_router(recommender_router)

# Stage 5: what-if disruption simulator
app.include_router(simulation_router)

# Stage 6: dashboard overview data
app.include_router(overview_router)


# ------------------------------------------------------------- endpoints ----
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe."""
    loaded = bool(getattr(app.state, "model_loaded", False))
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/api/predict-risk", response_model=RiskResponse)
def predict_risk_endpoint(
    body: ShipmentFeatures,
    threshold: float = Query(
        default=0.5, gt=0, lt=1, description="Decision threshold for predicted_disruption"
    ),
) -> RiskResponse:
    """Predict disruption risk for a single shipment.

    Returns risk_probability (0-100), risk_level (LOW/MEDIUM/HIGH),
    predicted_disruption and human-readable top_risk_factors.
    """
    try:
        result = predict_risk(body.model_dump(exclude_none=True), threshold=threshold)
    except ValueError as exc:  # defensive: schema should already prevent this
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RiskResponse(**result)


@app.post("/api/predict-risk/batch", response_model=list[RiskResponse])
def predict_risk_batch_endpoint(
    body: list[ShipmentFeatures],
    threshold: float = Query(
        default=0.5, gt=0, lt=1, description="Decision threshold for predicted_disruption"
    ),
) -> list[RiskResponse]:
    """Predict disruption risk for many shipments in one model pass."""
    if not body:
        raise HTTPException(status_code=422, detail="body must contain at least one shipment")
    if len(body) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=f"batch limited to {MAX_BATCH_SIZE} shipments")
    payloads = [item.model_dump(exclude_none=True) for item in body]
    results = predict_risk_batch(payloads, threshold=threshold)
    return [RiskResponse(**r) for r in results]
