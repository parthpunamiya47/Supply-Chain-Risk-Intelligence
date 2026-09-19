import { useEffect, useMemo, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, Legend } from 'recharts'
import { fetchGraph, fetchScenarios, runSimulation } from '../api'
import { Card, KpiCard, Badge, LoadingState, ErrorState, EmptyState, RiskBar } from '../components/ui'
import RiskNetwork from '../components/RiskNetwork'
import { RISK_COLORS, fmtPct, hexFor01 } from '../lib/risk'

const SCENARIO_META = {
  supplier_failure: { icon: '🏭', blurb: 'A supplier stops shipping entirely' },
  port_closure: { icon: '⚓', blurb: 'A seaport stops handling vessels' },
  transportation_disruption: { icon: '🚆', blurb: 'Rail/road lane degraded' },
  demand_spike: { icon: '📈', blurb: 'Demand surge drains warehouse stock' },
}

export default function SimulatorPage({ onSimulation, simulation }) {
  const [scenarios, setScenarios] = useState(null)
  const [graph, setGraph] = useState(null)
  const [selected, setSelected] = useState('supplier_failure')
  const [eventRisk, setEventRisk] = useState(0.95)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchScenarios().then((d) => {
      setScenarios(d.scenarios)
      const def = d.scenarios.find((s) => s.scenario === selected)
      if (def) setEventRisk(def.event_risk)
    }).catch(() => setScenarios([]))
    // the network overlay renders the real graph with the simulation result on top
    fetchGraph().then(setGraph).catch(() => setGraph(null))
  }, [])

  const pickScenario = (id) => {
    setSelected(id)
    const def = scenarios?.find((s) => s.scenario === id)
    if (def) setEventRisk(def.event_risk)
  }

  const run = () => {
    setRunning(true)
    setError(null)
    runSimulation({ scenario: selected, event_risk: Number(eventRisk) })
      .then((res) => onSimulation(res))
      .catch((e) => setError(e?.response?.data?.detail || e.message))
      .finally(() => setRunning(false))
  }

  const deltaData = useMemo(
    () =>
      (simulation?.affected_nodes || []).slice(0, 8).map((a) => ({
        node: a.node_id,
        before: a.before_risk * 100,
        after: a.after_risk * 100,
      })),
    [simulation],
  )

  return (
    <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
      <div className="space-y-4">
        <Card title="Scenario" subtitle="pick a disruption, then run it on a copy of the network">
          <div className="space-y-2">
            {(scenarios || Object.keys(SCENARIO_META)).map((s) => {
              const id = typeof s === 'string' ? s : s.scenario
              const info = typeof s === 'object' ? s : null
              const meta = SCENARIO_META[id] || {}
              const active = selected === id
              return (
                <button
                  key={id}
                  onClick={() => pickScenario(id)}
                  className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors ${
                    active ? 'border-slate-800 bg-slate-50 ring-1 ring-slate-300' : 'border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  <span className="text-lg">{meta.icon}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold text-slate-800">
                      {info?.label || meta.label || id.replaceAll('_', ' ')}
                    </span>
                    <span className="block truncate text-[11px] text-slate-500">
                      {info?.description || meta.blurb}
                    </span>
                  </span>
                  {active && <Badge tone="slate">selected</Badge>}
                </button>
              )
            })}
          </div>

          <label className="mt-4 block text-xs font-semibold text-slate-600">
            Event risk: <span className="num">{Math.round(eventRisk * 100)}%</span>
          </label>
          <input
            type="range" min="0.05" max="1" step="0.05" value={eventRisk}
            onChange={(e) => setEventRisk(Number(e.target.value))}
            className="mt-1 w-full accent-slate-800"
          />
          <button
            onClick={run}
            disabled={running}
            className="mt-4 w-full rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:opacity-50"
          >
            {running ? 'Simulating…' : 'Run simulation'}
          </button>
          {error && <p className="mt-2 text-xs text-rose-600">{error}</p>}
          <p className="mt-3 text-[11px] leading-snug text-slate-400">
            Simulations run on an in-memory copy of the network — the real graph is never modified.
          </p>
        </Card>
      </div>

      <div className="space-y-6">
        {!simulation && !running && (
          <Card title="No simulation yet">
            <EmptyState>
              Pick a scenario on the left and press <b>Run simulation</b> — results appear here and
              overlay the network on the Graph page.
            </EmptyState>
          </Card>
        )}
        {running && <Card><LoadingState label="Propagating risk through the network…" /></Card>}

        {simulation && !running && (
          <>
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <KpiCard label="Scenario" value={<span className="text-lg">{simulation.scenario_label}</span>} sub={`target: ${simulation.target_node} @ ${fmtPct(simulation.event_risk * 100, 0)}`} />
              <KpiCard label="Nodes affected" value={simulation.summary.nodes_affected} tone="amber" />
              <KpiCard label="Customers hit" value={simulation.summary.customers_affected} tone="rose" />
              <KpiCard label="Max risk jump" value={`+${fmtPct(simulation.summary.max_delta * 100, 1)}`} tone="rose" />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Before vs after risk" subtitle="top affected nodes (percent scale)">
                {deltaData.length ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={deltaData} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="node" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 11 }} unit="%" />
                      <Tooltip formatter={(v) => fmtPct(v, 1)} />
                      <Legend formatter={(v) => <span className="text-xs text-slate-600">{v}</span>} />
                      <Bar dataKey="before" name="before" fill="#94a3b8" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="after" name="after" radius={[3, 3, 0, 0]}>
                        {deltaData.map((d) => (
                          <Cell key={d.node} fill={hexFor01(d.after / 100)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyState>No downstream nodes affected</EmptyState>
                )}
              </Card>

              <Card title="Affected nodes" subtitle="before → after, sorted by delta">
                <ul className="thin-scroll max-h-[280px] space-y-2 overflow-y-auto pr-1">
                  {simulation.affected_nodes.map((a) => {
                    const deltaPct = a.delta * 100
                    return (
                      <li key={a.node_id} className="rounded-lg border border-slate-200 p-2.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate text-sm font-medium text-slate-800">{a.name}</span>
                          <span className="num shrink-0 text-xs font-bold text-rose-600">
                            +{deltaPct.toFixed(1)} pts
                          </span>
                        </div>
                        <div className="num mt-1 flex items-center gap-2 text-[11px] text-slate-500">
                          <span>{fmtPct(a.before_risk * 100, 1)}</span>
                          <span className="text-slate-300">→</span>
                          <span className="font-semibold" style={{ color: hexFor01(a.after_risk) }}>
                            {fmtPct(a.after_risk * 100, 1)}
                          </span>
                          <span className="ml-auto">{a.hops} hop{a.hops === 1 ? '' : 's'}</span>
                        </div>
                        <div className="mt-1.5"><RiskBar percent={a.after_risk * 100} /></div>
                      </li>
                    )
                  })}
                </ul>
              </Card>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Propagation paths" subtitle="dominant route from the event to each affected node">
                <ul className="thin-scroll max-h-[240px] space-y-1.5 overflow-y-auto pr-1 text-xs">
                  {simulation.propagation_paths.map((p) => (
                    <li key={p.node_id} className="flex items-center gap-1.5 rounded border border-slate-100 bg-slate-50 px-2.5 py-1.5">
                      {p.dominant_path.map((id, i) => (
                        <span key={i} className="flex items-center gap-1.5">
                          {i > 0 && <span className="text-slate-300">→</span>}
                          <span className="num font-semibold text-slate-700">{id}</span>
                        </span>
                      ))}
                    </li>
                  ))}
                </ul>
              </Card>

              <Card title="Recommended mitigations" subtitle="from the Stage 4 rule engine">
                <ul className="space-y-2.5">
                  {simulation.recommendations.map((r) => (
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
            </div>

            {graph && (
              <Card
                title="Network impact overlay"
                subtitle="indigo edges + 'simulated' badges mark affected nodes"
                className="overflow-hidden"
                bodyClassName="!p-0 h-[380px]"
              >
                <RiskNetwork graph={graph} simulation={simulation} />
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}
