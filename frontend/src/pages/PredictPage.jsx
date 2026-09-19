import { useState } from 'react'
import { predictRisk, recommendForShipment } from '../api'
import { Card, Badge, LoadingState, ErrorState, RiskBar } from '../components/ui'
import { RISK_COLORS, band, fmtPct, fmtNum } from '../lib/risk'

const FIELDS = [
  { name: 'supplier_reliability', label: 'Supplier reliability', min: 0, max: 100, step: 1, unit: '/100', def: 80 },
  { name: 'previous_delays', label: 'Previous delays (6 mo)', min: 0, max: 14, step: 1, unit: 'delays', def: 3 },
  { name: 'average_lead_time', label: 'Average lead time', min: 1, max: 45, step: 0.5, unit: 'days', def: 12 },
  { name: 'demand', label: 'Demand', min: 0, max: 60000, step: 100, unit: 'units', def: 5000 },
  { name: 'inventory_level', label: 'Inventory level', min: 0, max: 200000, step: 100, unit: 'units', def: 8000 },
  { name: 'weather_risk', label: 'Weather risk', min: 0, max: 10, step: 0.1, unit: '/10', def: 3 },
  { name: 'transportation_risk', label: 'Transportation risk', min: 0, max: 10, step: 0.1, unit: '/10', def: 3 },
  { name: 'distance', label: 'Distance', min: 0, max: 12000, step: 50, unit: 'km', def: 1500 },
  { name: 'supplier_capacity', label: 'Supplier capacity', min: 0, max: 200000, step: 500, unit: 'units/mo', def: 25000 },
  { name: 'shipment_size', label: 'Shipment size', min: 0, max: 60000, step: 100, unit: 'units', def: 4500 },
  { name: 'historical_disruptions', label: 'Historical disruptions (yr)', min: 0, max: 7, step: 1, unit: 'events', def: 1 },
]

const PRESETS = {
  'Typical lane': { supplier_reliability: 85, previous_delays: 2, average_lead_time: 12, demand: 5000, inventory_level: 8000, weather_risk: 3, transportation_risk: 3, distance: 1500, supplier_capacity: 25000, shipment_size: 4500, historical_disruptions: 1 },
  'Fragile supplier': { supplier_reliability: 50, previous_delays: 12, average_lead_time: 26, demand: 9000, inventory_level: 2200, weather_risk: 5.5, transportation_risk: 6, distance: 3800, supplier_capacity: 15000, shipment_size: 8000, historical_disruptions: 5 },
  'Storm + risky route': { supplier_reliability: 78, previous_delays: 4, average_lead_time: 18, demand: 6000, inventory_level: 5000, weather_risk: 9, transportation_risk: 8.5, distance: 6200, supplier_capacity: 30000, shipment_size: 5500, historical_disruptions: 2 },
  'Stockout pressure': { supplier_reliability: 88, previous_delays: 1, average_lead_time: 10, demand: 45000, inventory_level: 900, weather_risk: 3, transportation_risk: 3.5, distance: 900, supplier_capacity: 90000, shipment_size: 42000, historical_disruptions: 0 },
}

const toNum = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Number(v)]))

export default function PredictPage() {
  const [form, setForm] = useState(PRESETS['Typical lane'])
  const [threshold, setThreshold] = useState(0.5)
  const [result, setResult] = useState(null)
  const [recs, setRecs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (name, value) => setForm((f) => ({ ...f, [name]: value }))
  const applyPreset = (name) => setForm(PRESETS[name])

  const submit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const [prediction, recommendation] = await Promise.all([
        predictRisk(toNum(form)),
        recommendForShipment(toNum(form)).catch(() => null),
      ])
      setResult(prediction)
      setRecs(recommendation)
    } catch (err) {
      const detail = err?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : 'Validation failed — check the highlighted ranges.')
    } finally {
      setLoading(false)
    }
  }

  const c = result ? RISK_COLORS[band(result.risk_probability)] : null

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_420px]">
      <Card title="Shipment risk scoring" subtitle="11 model features — same schema the backend validates">
        <form onSubmit={submit} className="space-y-5">
          <div className="flex flex-wrap gap-2">
            {Object.keys(PRESETS).map((p) => (
              <button key={p} type="button" onClick={() => applyPreset(p)}
                className="rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-600 hover:border-slate-400 hover:bg-slate-50">
                {p}
              </button>
            ))}
          </div>

          <div className="grid gap-x-6 gap-y-4 md:grid-cols-2">
            {FIELDS.map((f) => (
              <div key={f.name}>
                <label className="flex items-baseline justify-between text-xs font-medium text-slate-600">
                  {f.label}
                  <span className="num text-slate-900">
                    {fmtNum(form[f.name])} <span className="text-slate-400">{f.unit}</span>
                  </span>
                </label>
                <input
                  type="range" min={f.min} max={f.max} step={f.step} value={form[f.name]}
                  onChange={(e) => set(f.name, e.target.value)}
                  className="mt-1 w-full accent-slate-800"
                />
              </div>
            ))}
          </div>

          <div className="flex items-center gap-4 border-t border-slate-100 pt-4">
            <label className="text-xs font-medium text-slate-600">
              Decision threshold: <span className="num">{threshold.toFixed(2)}</span>
            </label>
            <input type="range" min="0.1" max="0.9" step="0.05" value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-40 accent-slate-800" />
            <button type="submit" disabled={loading}
              className="ml-auto rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-50">
              {loading ? 'Scoring…' : 'Predict risk'}
            </button>
          </div>
        </form>
      </Card>

      <div className="space-y-6">
        {loading && <Card><LoadingState label="Querying the model…" /></Card>}
        {error && !loading && <ErrorState message={error} />}

        {result && !loading && (
          <>
            <Card title="Prediction" subtitle="Random Forest · Stage 1 model">
              <div className="flex items-end justify-between">
                <div>
                  <p className={`num text-5xl font-bold tracking-tight ${c.text}`}>
                    {fmtPct(result.risk_probability)}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">disruption probability</p>
                </div>
                <Badge tone={band(result.risk_probability)} className="!text-sm">{result.risk_level}</Badge>
              </div>
              <div className="mt-4"><RiskBar percent={result.risk_probability} /></div>
              <p className="mt-3 text-xs text-slate-500">
                Predicted disruption: <b>{String(result.predicted_disruption)}</b> at threshold {threshold.toFixed(2)}
              </p>
            </Card>

            <Card title="Key risk factors" subtitle="shipment vs healthy-shipment median">
              {result.top_risk_factors?.length ? (
                <ul className="space-y-2">
                  {result.top_risk_factors.map((f) => (
                    <li key={f} className="flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                      <span className="mt-0.5 text-rose-500">▲</span>{f}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-slate-500">
                  No factor stands out — every feature sits within normal variation of healthy shipments.
                </p>
              )}
            </Card>

            {recs?.recommendations?.length > 0 && (
              <Card title="Recommended actions" subtitle="Stage 4 rule engine">
                <ul className="space-y-2.5">
                  {recs.recommendations.map((r) => (
                    <li key={r.rule_id} className="rounded-lg border border-slate-200 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-semibold text-slate-800">{r.action}</span>
                        <Badge tone={r.priority}>{r.priority}</Badge>
                      </div>
                      <p className="mt-1 text-xs leading-snug text-slate-500">{r.reason}</p>
                      <p className="num mt-1 text-[11px] font-medium text-emerald-700">
                        potential risk reduction ≈ {fmtPct(r.risk_reduction)} · cost {r.cost_estimate}
                      </p>
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </>
        )}

        {!result && !loading && !error && (
          <Card title="Prediction">
            <p className="text-sm text-slate-500">
              Adjust the sliders (or pick a preset) and press <b>Predict risk</b> — the real trained
              model scores this shipment live. Recommended actions come from the rule engine.
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}
