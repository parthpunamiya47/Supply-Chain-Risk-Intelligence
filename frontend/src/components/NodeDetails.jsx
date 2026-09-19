import { useEffect, useState } from 'react'
import { predictRisk, recommendForShipment } from '../api'
import { Badge, RiskBar, LoadingState, EmptyState } from './ui'
import { RISK_COLORS, bandOf01, band, fmtPct, fmtNum } from '../lib/risk'

const PROFILE_BY_TYPE = {
  supplier: { supplier_reliability: 78, previous_delays: 4, average_lead_time: 20, demand: 9000, inventory_level: 4000, weather_risk: 4, transportation_risk: 5, distance: 5000, supplier_capacity: 12000, shipment_size: 8000, historical_disruptions: 2 },
  factory: { supplier_reliability: 78, previous_delays: 4, average_lead_time: 15, demand: 6000, inventory_level: 5000, weather_risk: 4, transportation_risk: 4, distance: 2000, supplier_capacity: 30000, shipment_size: 5000, historical_disruptions: 2 },
  port: { supplier_reliability: 80, previous_delays: 3, average_lead_time: 25, demand: 7000, inventory_level: 9000, weather_risk: 5.5, transportation_risk: 5.5, distance: 8000, supplier_capacity: 40000, shipment_size: 6000, historical_disruptions: 1 },
  route: { supplier_reliability: 82, previous_delays: 3, average_lead_time: 24, demand: 6500, inventory_level: 8000, weather_risk: 5, transportation_risk: 5.8, distance: 7000, supplier_capacity: 35000, shipment_size: 5500, historical_disruptions: 1 },
  warehouse: { supplier_reliability: 88, previous_delays: 2, average_lead_time: 10, demand: 4500, inventory_level: 5000, weather_risk: 3, transportation_risk: 3, distance: 900, supplier_capacity: 25000, shipment_size: 4000, historical_disruptions: 1 },
  customer: { supplier_reliability: 90, previous_delays: 1, average_lead_time: 8, demand: 3000, inventory_level: 2500, weather_risk: 2, transportation_risk: 2, distance: 400, supplier_capacity: 20000, shipment_size: 2800, historical_disruptions: 0 },
}

// Translate a graph node into representative shipment features so the real
// ML model can score it. The node's 0-1 baseline risk modulates the profile
// within a LOW->HIGH span (documented proxy, no invented API data).
function nodeToFeatures(node) {
  const base = PROFILE_BY_TYPE[node.type] || PROFILE_BY_TYPE.warehouse
  const r = node.risk ?? 0 // 0..1
  return {
    ...base,
    supplier_reliability: Math.round(Math.max(30, base.supplier_reliability - r * 35)),
    previous_delays: Math.min(14, Math.round(base.previous_delays + r * 8)),
    historical_disruptions: Math.min(7, Math.round(base.historical_disruptions + r * 5)),
    weather_risk: Math.min(10, +(base.weather_risk + r * 3).toFixed(1)),
    transportation_risk: Math.min(10, +(base.transportation_risk + r * 3).toFixed(1)),
    inventory_level: Math.round(base.inventory_level * (1 - 0.5 * r)),
  }
}

export default function NodeDetails({ node, graph, onClose }) {
  const [ml, setMl] = useState(null)
  const [mlLoading, setMlLoading] = useState(false)
  const [mlError, setMlError] = useState(null)
  const [recs, setRecs] = useState(null)
  const [recsLoading, setRecsLoading] = useState(false)

  useEffect(() => {
    if (!node) return
    let cancelled = false
    setMl(null); setMlError(null); setRecs(null)
    setMlLoading(true)
    const features = nodeToFeatures(node)
    ;(async () => {
      try {
        const [prediction, recommendation] = await Promise.all([
          predictRisk(features),
          recommendForShipment(features).catch(() => null),
        ])
        if (!cancelled) {
          setMl({ ...prediction, features })
          setRecs(recommendation)
        }
      } catch (e) {
        if (!cancelled) setMlError(e?.response?.data?.detail || e.message)
      } finally {
        if (!cancelled) setMlLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [node?.id])

  if (!node) return null
  const c = RISK_COLORS[bandOf01(node.risk)]
  const mlBand = ml ? band(ml.risk_probability) : null

  return (
    <aside className="thin-scroll flex h-full w-[340px] shrink-0 flex-col overflow-y-auto border-l border-slate-200 bg-white">
      <div className="flex items-start justify-between gap-2 border-b border-slate-100 px-5 py-4">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">{node.type}</p>
          <h3 className="text-sm font-semibold text-slate-900">{node.label}</h3>
          {node.location && <p className="text-xs text-slate-500">{node.location}</p>}
        </div>
        <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600" aria-label="Close">✕</button>
      </div>

      <div className="space-y-4 px-5 py-4">
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="font-medium text-slate-600">Baseline risk</span>
            <span className={`num font-bold ${c.text}`}>{fmtPct(node.risk * 100, 0)}</span>
          </div>
          <RiskBar percent={node.risk * 100} />
          {node.capacity != null && (
            <p className="mt-2 text-xs text-slate-500">Capacity: <span className="num font-medium text-slate-700">{fmtNum(node.capacity)}</span> units</p>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Model risk probability</p>
          {mlLoading ? (
            <LoadingState label="Scoring with Stage 1 model…" />
          ) : mlError ? (
            <p className="mt-1 text-xs text-rose-600">{mlError}</p>
          ) : ml ? (
            <>
              <div className="mt-1 flex items-baseline gap-2">
                <span className={`num text-2xl font-bold ${RISK_COLORS[mlBand].text}`}>
                  {fmtPct(ml.risk_probability)}
                </span>
                <Badge tone={mlBand}>{mlBand}</Badge>
              </div>
              <p className="mt-1 text-[11px] leading-snug text-slate-500">
                Predicted disruption probability from the Random Forest using
                representative {node.type} features. Simulated: {String(ml.predicted_disruption)}.
              </p>
              {ml.top_risk_factors?.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {ml.top_risk_factors.map((f) => (
                    <li key={f} className="flex items-start gap-1.5 text-xs text-slate-700">
                      <span className="mt-0.5 text-rose-500">▲</span>{f}
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : null}
        </div>

        {recs?.recommendations?.length > 0 && (
          <div>
            <p className="mb-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-400">Mitigations</p>
            <ul className="space-y-2">
              {recs.recommendations.slice(0, 3).map((r) => (
                <li key={r.rule_id} className="rounded-lg border border-slate-200 p-2.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-slate-800">{r.action}</span>
                    <Badge tone={r.priority}>{r.priority}</Badge>
                  </div>
                  <p className="mt-1 text-[11px] leading-snug text-slate-500">{r.reason}</p>
                  <p className="mt-1 num text-[11px] font-medium text-emerald-700">
                    potential risk reduction ≈ {fmtPct(r.risk_reduction, 1)}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </aside>
  )
}
