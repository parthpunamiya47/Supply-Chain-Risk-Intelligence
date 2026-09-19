import axios from 'axios'

// Relative base URL: Vite's dev proxy (and any reverse proxy in production)
// forwards /api to the FastAPI backend. No fake responses anywhere - if the
// backend is down, the UI shows the error state honestly.
const api = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL}/api`,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

// ---- overview -------------------------------------------------------------
export const fetchOverview = () => api.get('/overview').then((r) => r.data)

// ---- graph ----------------------------------------------------------------
export const fetchGraph = () => api.get('/graph').then((r) => r.data)

// ---- risk prediction --------------------------------------------------------
export const predictRisk = (features, threshold) =>
  api
    .post('/predict-risk', features, {
      params: threshold != null ? { threshold } : undefined,
    })
    .then((r) => r.data)

export const recommendForShipment = (features, threshold) =>
  api
    .post('/recommendations', { features, threshold }, { params: {} })
    .then((r) => r.data)

// ---- simulation -------------------------------------------------------------
export const fetchScenarios = () =>
  api.get('/simulation/scenarios').then((r) => r.data)

export const runSimulation = (body) =>
  api.post('/simulation', body).then((r) => r.data)

// ---- propagation (node drill-down) -----------------------------------------
export const propagateFromNode = (nodeId, riskProbability) =>
  api
    .post('/graph/propagate-risk', { node_id: nodeId, risk_probability: riskProbability })
    .then((r) => r.data)
