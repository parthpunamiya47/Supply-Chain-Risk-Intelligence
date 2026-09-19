"""SupplyChain Sentinel - Stage 2: Pydantic schemas for the risk API.

Validation layers (Task 3):
  - required fields & types: enforced by the Pydantic model itself
  - numeric ranges:          Field(gt=..., le=...) constraints
  - invalid inputs:          FastAPI turns these into 422 with structured details

The response model matches the Stage 1 `predict_risk()` contract exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ShipmentFeatures(BaseModel):
    """Request body for POST /api/predict-risk."""

    # supplier_id is optional metadata (ignored by the model, kept for logs)
    supplier_id: str | None = Field(default=None, max_length=64)

    supplier_reliability: float = Field(
        ..., ge=0, le=100, description="0-100 score; higher = more reliable"
    )
    previous_delays: int = Field(..., ge=0, le=365, description="delays in past 6 months")
    average_lead_time: float = Field(..., gt=0, le=365, description="days")
    demand: int = Field(..., ge=0, description="units demanded at destination")
    inventory_level: int = Field(..., ge=0, description="units in stock at destination")
    weather_risk: float = Field(..., ge=0, le=10, description="0-10 scale")
    transportation_risk: float = Field(..., ge=0, le=10, description="0-10 scale")
    distance: float = Field(..., ge=0, le=100_000, description="km")
    supplier_capacity: int = Field(..., ge=0, description="units/month")
    shipment_size: int = Field(..., ge=0, description="units")
    historical_disruptions: int = Field(..., ge=0, le=1000, description="past year")

    # threshold is supplied via query param, not body; body only carries features
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                }
            ]
        }
    }

    @field_validator("supplier_id")
    @classmethod
    def _strip_supplier_id(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("supplier_id must not be blank")
        return v.strip() if v else v


class RiskFactor(BaseModel):
    """One human-readable driver behind a prediction."""

    factor: str
    feature: str
    value: float
    healthy_median: float
    deviation_iqr: float


class RiskResponse(BaseModel):
    """Response contract (identical shape to Stage 1's predict_risk output)."""

    risk_probability: float = Field(..., ge=0, le=100, examples=[82.4])
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$", examples=["HIGH"])
    predicted_disruption: bool
    top_risk_factors: list[str]


class DetailedRiskResponse(RiskResponse):
    """Adds structured risk factors + model metadata for debugging/UX."""

    top_risk_factors_detailed: list[RiskFactor] = []
    model_version: str = "stage1-rf-v1"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    model_version: str = "stage1-rf-v1"
