import { useEffect, useState } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, LineChart, Line, Legend,
} from 'recharts'
import { fetchOverview } from '../api'
import { Card, KpiCard, Badge, RiskBar, LoadingState, ErrorState, EmptyState } from '../components/ui'
import { RISK_COLORS, fmtPct, fmtNum } from '../lib/risk'

export default function OverviewPage() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setError(null)
    fetchOverview()
      .then(setData)
      .catch((e) => setError(e?.message || 'Request failed'))
      .finally(() => setLoading(false))
  }
  useEffect(load, [])

  if (loading) return <LoadingState label="Loading dashboard KPIs…" />
  if (error) return <ErrorState message={error} onRetry={load} />
  if (!data) return null

  const dist = ['HIGH', 'MEDIUM', 'LOW']
    .map((k) => ({ name: k, value: data.risk_distribution?.[k] ?? 0, fill: RISK_COLORS[k].hex }))
    .filter((d) => d.value > 0)
  const predDist = ['HIGH', 'MEDIUM', 'LOW']
    .map((k) => ({ name: k, value: data.prediction_distribution?.[k] ?? 0 }))
    .filter((d) => d.value > 0)

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <KpiCard label="Total suppliers" value={data.total_suppliers} sub={`${data.graph_nodes} network nodes`} />
        <KpiCard
          label="High-risk nodes"
          value={data.high_risk_nodes}
          tone={data.high_risk_nodes > 0 ? 'rose' : 'emerald'}
          sub="graph baseline ≥ 70"
        />
        <KpiCard
          label="Active alerts"
          value={data.alerts?.length ?? 0}
          tone={(data.alerts?.length ?? 0) > 0 ? 'amber' : 'emerald'}
          sub="nodes ≥ 40 baseline"
        />
        <KpiCard
          label="At-risk shipments"
          value={data.at_risk_shipments}
          tone="amber"
          sub={data.sample_size ? `of ${data.sample_size} model-scored` : 'model scoring pending'}
        />
        <KpiCard label="Average risk" value={fmtPct(data.average_risk)} sub="model, scored sample" tone="sky" />
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Graph risk distribution" subtitle="baseline band per network node">
          {dist.length ? (
            <ResponsiveContainer width="100%" height={230}>
              <PieChart>
                <Pie data={dist} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={3}>
                  {dist.map((d) => <Cell key={d.name} fill={d.fill} />)}
                </Pie>
                <Tooltip />
                <Legend iconType="circle" formatter={(v) => <span className="text-xs text-slate-600">{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState>No nodes</EmptyState>
          )}
        </Card>

        <Card title="Model risk distribution" subtitle={data.sample_size ? `${data.sample_size} shipments scored by the Random Forest` : 'model not trained yet'}>
          {predDist.length ? (
            <ResponsiveContainer width="100%" height={230}>
              <PieChart>
                <Pie data={predDist} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={3}>
                  {predDist.map((d) => <Cell key={d.name} fill={RISK_COLORS[d.name].hex} />)}
                </Pie>
                <Tooltip />
                <Legend iconType="circle" formatter={(v) => <span className="text-xs text-slate-600">{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState>Train the model to populate this chart</EmptyState>
          )}
        </Card>

        <Card title="Disruption trend" subtitle="realized weekly rate from shipment history">
          {data.risk_trend?.length ? (
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={data.risk_trend} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" />
                <Tooltip formatter={(v) => fmtPct(v)} />
                <Line type="monotone" dataKey="disruption_rate_pct" stroke="#e11d48" strokeWidth={2} dot={{ r: 3 }} name="disruption rate" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState>No trend data</EmptyState>
          )}
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Supplier performance" subtitle="top-10 suppliers by volume — realized disruption rate (dataset)">
          {data.supplier_performance?.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={data.supplier_performance} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="supplier_id" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} unit="%" />
                <Tooltip formatter={(v, n) => (n === 'disruption_rate_pct' ? fmtPct(v) : v)} />
                <Bar dataKey="disruption_rate_pct" name="disruption rate" radius={[4, 4, 0, 0]}>
                  {data.supplier_performance.map((s) => (
                    <Cell key={s.supplier_id} fill={RISK_COLORS[s.disruption_rate_pct >= 40 ? 'HIGH' : s.disruption_rate_pct >= 27 ? 'MEDIUM' : 'LOW'].hex} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState>Generate the dataset to populate this chart</EmptyState>
          )}
        </Card>

        <Card title="Active alerts" subtitle="network nodes above the monitoring threshold">
          {data.alerts?.length ? (
            <ul className="thin-scroll max-h-[260px] space-y-2.5 overflow-y-auto pr-1">
              {data.alerts.map((a) => (
                <li key={a.id} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium text-slate-800">{a.name}</span>
                    <Badge tone={a.severity}>{a.severity}</Badge>
                  </div>
                  <p className="mt-0.5 text-xs text-slate-500">{a.message}</p>
                  <div className="mt-2"><RiskBar percent={a.risk * 100} /></div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState>No active alerts — all nodes below threshold</EmptyState>
          )}
        </Card>
      </div>
    </div>
  )
}
