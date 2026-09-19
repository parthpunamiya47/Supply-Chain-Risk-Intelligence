# SupplyChain Sentinel — Stages 1-6: ML · API · Graph · Recommendations · Simulator · Dashboard

Synthetic-data-driven disruption-risk model for shipments. Stage 1 covers the
data + ML foundation: dataset generation, EDA, preprocessing, a Random Forest
classifier, evaluation, explainability, and a prediction API. Stage 2 wraps the
trained model in a **FastAPI backend** with validation, CORS, and tests. Stage 3
adds the **supply-chain dependency graph** with explainable downstream **risk
propagation**. Stage 4 adds a **rule-based mitigation recommendation engine**
connected to both the ML prediction and the graph. Stage 5 adds the **what-if
disruption simulator** (runs on a graph copy; the real network is never
modified). Stage 6 adds the **React dashboard** (Vite + Tailwind + React Flow +
Recharts) connected to the live backend — no mock data anywhere.

> ⚠️ All data is **synthetic**. Metrics describe the synthetic dataset's baked-in
> relationships and make **no claims about real-world performance**.

## Project structure

```
/frontend                          # Stage 6 React dashboard (Vite + Tailwind + React Flow + Recharts)
  src/api.js                         # axios client for every backend endpoint
  src/App.jsx                        # enterprise shell: sidebar nav, API status, routing
  src/pages/OverviewPage.jsx         # KPIs, risk charts, supplier performance, alerts
  src/pages/GraphPage.jsx            # React Flow network + node drill-down panel
  src/pages/PredictPage.jsx          # 11-feature scoring form (real model calls)
  src/pages/SimulatorPage.jsx        # what-if scenarios with before/after overlay
  src/components/RiskNetwork.jsx     # risk-colored React Flow nodes + simulation overlay
  src/components/NodeDetails.jsx     # node panel: baseline risk, ML probability, factors, mitigations
/backend
  main.py                             # FastAPI app: /api/health, /api/predict-risk(+ /batch), CORS
  overview.py / overview_router.py    # Stage 6: GET /api/overview (real dashboard KPIs)
  schemas.py                          # Pydantic request/response models (validation)
  graph.py                            # Stage 3 demo network (~14 nodes) + React Flow serializer
  propagation.py                      # Stage 3 explainable risk propagation (documented formula)
  graph_schemas.py                    # Stage 3 Pydantic models for the graph API
  graph_router.py                     # Stage 3 endpoints: /api/graph, /api/graph/propagate-risk
  recommendations.py                  # Stage 4 rule engine (6 rules, documented thresholds)
  recommender_schemas.py              # Stage 4 Pydantic models
  recommender_router.py               # Stage 4 endpoint: POST /api/recommendations
  simulation.py                       # Stage 5 what-if scenarios on a graph COPY
  simulation_schemas.py               # Stage 5 Pydantic models
  simulation_router.py                # Stage 5 endpoints: POST /api/simulation, GET .../scenarios
  tests/test_api.py                   # pytest API tests (valid / invalid / health / CORS)
  tests/test_graph.py                 # graph structure + 3 propagation scenarios (formula-checked)
  tests/test_recommendations.py       # recommendation scenarios (every rule covered)
  tests/test_simulation.py            # 4 what-if scenarios + copy-integrity checks
/data
  supply_chain_shipments.csv          # generated dataset (~2,500 rows)
  analysis/                           # EDA plots + tables (missing values, class balance, ...)
/ml
  generate_dataset.py                 # Task 1 - synthetic dataset generator
  analyze_data.py                     # Task 2 - EDA: missing, duplicates, distributions, correlations, imbalance
  preprocess.py                       # Task 3 - reusable ColumnTransformer + leak-free train/test split
  train.py                            # Task 4 - Random Forest training (+ joblib persistence)
  evaluate.py                         # Task 5 - precision/recall/F1, confusion matrix, ROC-AUC, threshold scan
  explain.py                          # Task 6 - global importances + per-shipment top_risk_factors
  predict.py                          # Task 7 - predict_risk() API + CLI + sample shipments
  run_pipeline.py                     # runs Tasks 1-7 end to end
/models
  random_forest_pipeline.joblib       # preprocessor + classifier in one artifact
  feature_importances.json            # permutation importances
  training_metadata.json              # training config + test metrics
  metrics.json                        # detailed evaluation metrics
  rf_evaluation_report.txt            # human-readable evaluation report
  confusion_matrix.png / roc_curve.png / precision_recall_curve.png
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

# one command for the full Stage 1 flow:
python ml/run_pipeline.py

# ...or step by step:
python ml/generate_dataset.py --rows 2500
python ml/analyze_data.py
python ml/train.py
python ml/evaluate.py            # standalone re-evaluation of the saved model
python ml/explain.py
python ml/predict.py --samples
```

## Backend — exact start commands (Stage 2)

> The API needs the trained model. If `models/random_forest_pipeline.joblib` is
> missing, run `python ml/train.py` first.

From the **project root**:

```bash
# Windows (Git Bash / PowerShell with .venv activated)
.venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000

# macOS / Linux
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000

# alternative: activated venv
source .venv/Scripts/activate     # Windows Git Bash (macOS/Linux: source .venv/bin/activate)
uvicorn backend.main:app --reload --port 8000
```

Then open:

- Swagger UI: <http://127.0.0.1:8000/docs> (interactive try-it-out)
- Health check:

```bash
curl http://127.0.0.1:8000/api/health
# {"status":"healthy","model_loaded":true,"model_version":"stage1-rf-v1"}
```

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/api/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
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
    "historical_disruptions": 5
  }'
```

Optional: `?threshold=0.35` query param changes the `predicted_disruption`
cut-off. Batch endpoint: `POST /api/predict-risk/batch` with a JSON array
(max 500 shipments).

Run the API tests:

```bash
.venv/Scripts/python -m pytest backend/tests -q      # Windows
.venv/bin/python -m pytest backend/tests -q          # macOS/Linux
# 55 tests: 13 risk API + 14 graph + 12 recommendations + 14 simulation + 2 overview
```

## Frontend — exact start commands (Stage 6)

Start the backend first, then the dev server (Vite proxies `/api` to it):

```bash
# terminal 1 — backend (from project root)
.venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm install            # first time only
npm run dev            # http://localhost:5173
```

Production build: `cd frontend && npm run build` (output in `frontend/dist/`).

### Dashboard sections

1. **Overview** — KPI cards (total suppliers, high-risk nodes, active alerts,
   at-risk shipments, average risk) + risk distribution, model distribution,
   disruption trend and supplier performance charts + alert list. All computed
   from the real dataset/model/graph via `GET /api/overview`.
2. **Supply Chain Graph** — React Flow network (green/yellow/red risk nodes,
   tiered supplier→factory→warehouse→customer layout). Clicking a node shows
   name, type, baseline risk, capacity, the **ML risk probability** scored live
   by `POST /api/predict-risk` (representative per-type features), key risk
   factors and mitigations. Active simulations overlay `simulated` badges and
   indigo edges.
3. **Risk Prediction** — 11 sliders + presets → `POST /api/predict-risk` →
   risk %, level badge, factors, plus Stage 4 recommendations for the same
   shipment.
4. **Alert panel** — network nodes above the monitoring threshold (Overview +
   sidebar bell count).
5. **Recommendations** — rendered wherever relevant (prediction page, node
   panel, simulator) with action / reason / expected reduction / priority / cost.
6. **What-If Simulator** — Supplier Failure, Port Closure, Transportation
   Disruption, Demand Spike (event-risk slider, optional demand multiplier) →
   `POST /api/simulation` → before/after chart, affected-node list, propagation
   paths, recommendations and the network overlay. The simulation badge stays
   visible across pages and can be dismissed by starting the next run.

> All numbers come from the live backend. If an endpoint is down the UI shows
> an explicit error state — there are no fake API responses.

### API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness/readiness; returns model status |
| POST | `/api/predict-risk` | Single-shipment disruption risk |
| POST | `/api/predict-risk/batch` | Up to 500 shipments per call |
| GET | `/api/graph` | Full demo network as React Flow JSON |
| POST | `/api/graph/propagate-risk` | Propagate a risk event downstream |
| POST | `/api/recommendations` | Mitigation actions (shipment or graph event) |
| GET | `/api/simulation/scenarios` | Catalog of built-in what-if scenarios |
| POST | `/api/simulation` | Run a what-if disruption on a graph copy |

## Stage 3 — dependency graph & risk propagation

### Network

~14 nodes in tiers: 4 suppliers → 2 factories → (port + rail middle mile) →
3 warehouses → 3 customers. Each node carries `id, name, type, risk (baseline
0-1), capacity, location`; each edge a `relationship` (`supplies`,
`ships_via`, `stores_to`, `serves`) and a dependency `weight` (0-1 = fraction
of the target's input that depends on the source).

### Propagation formula (simple & explainable — no GNN)

For a risk event `R_s` at node `s`, every downstream node `n` receives:

```
propagated(n) = R_s × Π_path ( weight_e × DECAY )        over the path edges
DECAY = 0.7 per hop          MAX_HOPS = 4          prune contributions < 0.01
```

- **Dependency weight**: a 0.3-weight edge passes 30% of what a 1.0 edge would
  at the same distance.
- **Graph distance**: each extra hop multiplies by `DECAY`, so 1 hop away a
  node gets up to `0.7 × R_s`, 2 hops `0.49 × R_s` (times relative weights).
- **Multi-path**: a node's reported risk is the **maximum** single-path
  contribution (bounded by `R_s`); all alternative paths are returned too, so
  the dominant route is always visible.
- Baseline node risk is reported separately and never mixed into the
  propagated value.

Example: Supplier C at `0.85` → Factory 2 (`0.85 × 0.85 × 0.7 = 0.5057`) →
Rail (`× 0.6 × 0.7 = 0.2124`) → warehouses → customers (≤ 4 hops).

```bash
# graph as React Flow JSON
curl http://127.0.0.1:8000/api/graph

# propagate a supplier risk event
curl -X POST http://127.0.0.1:8000/api/graph/propagate-risk \
  -H "Content-Type: application/json" \
  -d '{"node_id": "sup_c", "risk_probability": 0.85}'
# -> {"source":"sup_c", "original_risk":0.85,
#     "affected_nodes":[{"node_id":"fac_2", "propagated_risk":0.5057,
#                        "propagation_path":["sup_c","fac_2"], ...}], ...}
```

The response contains, per affected node: `propagated_risk` (0-1),
`contribution` (same value in %), `hops`, the dominant `propagation_path`,
`path_edges` (with each edge's weight), `alternative_paths`, plus the node's
own `original_risk` baseline. `summary` reports how many nodes/customers were
affected and the maximum propagated risk.

## Stage 4 — mitigation recommendations (rule-based, no LLM)

`POST /api/recommendations` runs six explicit rules over a risk context built
from **either** shipment features (Stage 1 ML prediction) **or** a graph event
(Stage 3 propagation of `{"node_id", "risk_probability"}`). All thresholds live
in `backend/recommendations.py` and are printed here:

| Rule | Fires when | Action | Default priority | Cost |
|---|---|---|---|---|
| R1 | supplier risk > 70 **and** alternate supplier exists | Switch to alternate supplier | HIGH (≥85) / MEDIUM | HIGH |
| R2 | supplier risk > 70 **and** inventory < 0.5 × demand | Increase safety stock | HIGH | MEDIUM |
| R3 | transportation risk > 70 | Reroute via alternate lane/port | HIGH (≥85) / MEDIUM | MEDIUM |
| R4 | inventory < 0.5 × demand **and** demand > 5,000 | Expedite replenishment | MEDIUM | MEDIUM |
| R5 | average lead time > 21 days | Buffer the committed delivery date | LOW | LOW |
| R6 | overall risk 40–69 and no HIGH driver | Enhanced monitoring | MEDIUM | LOW |
| R0 | overall risk < 40 and nothing fired | No action required | NONE | NONE |

Each recommendation returns `action`, `reason` (traces back to the exact
threshold), `risk_reduction` (heuristic estimate tied to the model's global
feature importances - explicitly **not** a measured causal effect),
`cost_estimate` (qualitative NONE/LOW/MEDIUM/HIGH + note), `priority`, and
`details` with the trigger values.

```bash
curl -X POST http://127.0.0.1:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"features": {"supplier_reliability": 50.0, "previous_delays": 13, "average_lead_time": 30.0,
        "demand": 12000, "inventory_level": 1500, "weather_risk": 9.0, "transportation_risk": 9.0,
        "distance": 8000, "supplier_capacity": 9000, "shipment_size": 10000,
        "historical_disruptions": 6}}'
# -> recommendations sorted HIGH first: R3 reroute, R2 safety stock, R1 switch supplier, ...

curl -X POST http://127.0.0.1:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"node_id": "sup_c", "risk_probability": 0.85}'
# -> graph mode: propagates the event, assesses the most-affected downstream node
```

The response includes `context` (risk level, supplier/transport risk, inventory
ratio, top risk factors), the sorted `recommendations`, and a `summary`
(`rules_fired`, `highest_priority`, `total_potential_risk_reduction`).
Honesty note: `risk_reduction` values are order-of-magnitude heuristics for the
hackathon demo; treat them as relative rankings, not guarantees.

Live scenario walkthrough: `.venv/Scripts/python scripts/verify_stage4.py`

## Stage 5 — what-if disruption simulator

`POST /api/simulation` runs hypothetical scenarios **on an in-memory copy** of
the network (`build_graph().copy()`); the real graph is never mutated (asserted
by tests and a live before/after fingerprint).

Built-in scenarios (`GET /api/simulation/scenarios`, all parameters
overridable per request):

| Scenario | Default target | Event risk | Story | Overrides |
|---|---|---|---|---|
| `supplier_failure` | `sup_c` | 0.95 | Supplier stops shipping | supplier_risk 95 |
| `port_closure` | `port_lb` | 0.95 | Port stops handling vessels | transport 92, lead 30d |
| `transportation_disruption` | `route_rail` | 0.90 | Rail lane degraded | transport 90, lead 27d |
| `demand_spike` | `wh_2` | 0.55 | Demand surge drains stock | demand 18k, inv ratio 0.3 |

Pipeline per request: copy graph → apply event (raise target risk) → propagate
(Stage 3 formula) → before/after merge (`after = max(baseline, propagated)`,
the event node itself jumps to the event risk) → recommendations (Stage 4
engine, fed scenario-adjusted economics) → response.

```bash
curl -X POST http://127.0.0.1:8000/api/simulation \
  -H "Content-Type: application/json" \
  -d '{"scenario": "supplier_failure", "target_node": "sup_a", "event_risk": 0.8}'
# -> {"scenario": ..., "before_risk": {...}, "after_risk": {...},
#     "affected_nodes": [{"node_id": "fac_1", "before_risk": 0.10,
#                         "after_risk": 0.448, "delta": 0.348, "hops": 1}, ...],
#     "propagation_paths": [{"node_id": ..., "dominant_path": [...], ...}],
#     "recommendations": [{"rule_id": "R1", "action": "Switch to alternate supplier", ...}],
#     "summary": {"graph_modified": false, ...}}
```

Errors: unknown scenario / unknown `target_node` → 404; wrong node type for
the scenario, `event_risk` outside (0, 1], non-positive `demand_multiplier`
→ 422.

Live walkthrough: `.venv/Scripts/python scripts/verify_stage5.py`

- **Validation (Pydantic)**: all 11 numeric features are required; ranges are
  enforced (reliability 0-100, weather/transport risk 0-10, lead time > 0, etc.).
  Violations return `422` with the offending field names in `detail`.
- **CORS**: allowed dev origins default to `localhost:3000` (CRA) and
  `localhost:5173` (Vite). Set `CORS_ORIGINS="https://yourdomain,https://app.yourdomain"`
  to override in deployment.
- `supplier_id` is optional metadata - it is accepted for traceability but never
  fed to the model.

## Direct prediction API (used by the backend)

```python
from ml.predict import predict_risk

result = predict_risk({
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
})
# {
#   "risk_probability": 82.4,      # 0-100
#   "risk_level": "HIGH",          # LOW <40 | MEDIUM 40-69 | HIGH >=70
#   "predicted_disruption": true,  # class at 0.5 threshold (tunable)
#   "top_risk_factors": ["High previous delays", "Low supplier reliability", ...]
# }
```

CLI equivalents:

```bash
python ml/predict.py --samples
python ml/predict.py --input-json '{"supplier_reliability": 52, ...}'
python ml/predict.py --input-file shipment.json
```

## Modelling notes

- **Class imbalance**: disruptions are a minority (~25-30% of rows). The forest uses
  `class_weight="balanced"`; we report precision/recall/F1 and ROC-AUC rather than accuracy.
- **Leak-free design**: the test split is carved out *before* any fitting; the
  imputer is fitted on training data only and stored inside the joblib pipeline,
  so serving uses exactly the same preprocessing as training.
- **Thresholds**: the reported metrics use the 0.5 default; `models/rf_evaluation_report.txt`
  includes a precision/recall trade-off table if the backend wants stricter alerts.
- **Explainability**: global permutation importances (ROC-AUC scored) plus a local
  explanation that contrasts a shipment's risky features against the healthy
  (non-disrupted) median profile.

## Ideas beyond Stage 6

- `frontend/` is intentionally lean: auth, theming and i18n were skipped on purpose.
- `ml/preprocess.py` owns the canonical feature list - extend there if the schema grows.
- Candidate next steps: compound scenarios, persistence for simulation history, exports.
