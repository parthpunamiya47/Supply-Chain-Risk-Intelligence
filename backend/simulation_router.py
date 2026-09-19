"""SupplyChain Sentinel - Stage 5: simulation API router.

  GET  /api/simulation/scenarios  -> available demo scenarios
  POST /api/simulation            -> run a what-if scenario on a graph copy
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.simulation import SCENARIOS, run_simulation
from backend.simulation_schemas import (
    ScenarioInfo,
    ScenarioListResponse,
    SimulationRequest,
    SimulationResponse,
)

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.get("/scenarios", response_model=ScenarioListResponse)
def list_scenarios() -> ScenarioListResponse:
    """Catalog of built-in what-if scenarios."""
    return ScenarioListResponse(
        scenarios=[
            ScenarioInfo(
                scenario=sid,
                label=s["label"],
                description=s["description"],
                default_target=s["default_target"],
                target_type=s["target_type"],
                event_risk=s["event_risk"],
            )
            for sid, s in SCENARIOS.items()
        ]
    )


@router.post("", response_model=SimulationResponse)
def simulate(body: SimulationRequest) -> SimulationResponse:
    """Run a what-if disruption scenario on an in-memory copy of the network.

    The real graph is never modified - the simulation works on `g.copy()`.
    """
    try:
        result = run_simulation(
            body.scenario,
            target_node=body.target_node,
            event_risk=body.event_risk,
            demand_multiplier=body.demand_multiplier,
        )
    except KeyError as exc:
        # unknown scenario or unknown target node
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        # wrong node type for the scenario, bad event_risk / demand_multiplier
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SimulationResponse(**result)
