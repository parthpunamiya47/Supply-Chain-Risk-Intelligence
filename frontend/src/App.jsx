import { useEffect, useState } from 'react'
import OverviewPage from './pages/OverviewPage'
import GraphPage from './pages/GraphPage'
import PredictPage from './pages/PredictPage'
import SimulatorPage from './pages/SimulatorPage'
import { fetchOverview } from './api'
import { Badge } from './components/ui'

const NAV = [
  { id: 'overview', label: 'Overview', icon: '📊', desc: 'Network KPIs & trends' },
  { id: 'graph', label: 'Supply Chain Graph', icon: '🕸️', desc: 'Interactive dependency map' },
  { id: 'predict', label: 'Risk Prediction', icon: '🎯', desc: 'Score a shipment' },
  { id: 'simulator', label: 'What-If Simulator', icon: '🧪', desc: 'Run disruption scenarios' },
]

export default function App() {
  const [page, setPage] = useState('overview')
  const [simulation, setSimulation] = useState(null)
  const [health, setHealth] = useState('checking')
  const [alertCount, setAlertCount] = useState(null)

  useEffect(() => {
    fetchOverview()
      .then((d) => {
        setHealth('healthy')
        setAlertCount(d.alerts?.length ?? 0)
      })
      .catch(() => setHealth('down'))
  }, [page])

  return (
    <div className="flex min-h-screen">
      {/* sidebar */}
      <aside className="flex w-14 shrink-0 flex-col border-r border-slate-800 bg-slate-900 text-slate-300 md:w-60">
        <div className="flex items-center gap-2.5 px-3 py-5 md:px-5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-rose-500 to-amber-500 text-lg">🛡️</span>
          <div className="hidden md:block">
            <p className="text-sm font-bold leading-tight text-white">SupplyChain Sentinel</p>
            <p className="text-[10px] uppercase tracking-widest text-slate-500">Risk Control Center</p>
          </div>
        </div>

        <nav className="mt-2 flex-1 space-y-1 px-2 md:px-3">
          {NAV.map((item) => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              title={item.label}
              className={`flex w-full items-center gap-3 rounded-lg px-2.5 py-2.5 text-left transition-colors md:px-3 ${
                page === item.id ? 'bg-slate-800 text-white' : 'hover:bg-slate-800/60 hover:text-slate-100'
              }`}
            >
              <span className="shrink-0">{item.icon}</span>
              <span className="hidden min-w-0 md:block">
                <span className="block truncate text-sm font-medium">{item.label}</span>
                <span className="block truncate text-[10px] text-slate-500">{item.desc}</span>
              </span>
              {item.id === 'overview' && alertCount > 0 && (
                <span className="ml-auto rounded-full bg-rose-600 px-1.5 py-0.5 text-[10px] font-bold text-white">
                  {alertCount}
                </span>
              )}
            </button>
          ))}
        </nav>

        <div className="border-t border-slate-800 px-3 py-4 md:px-5">
          <div className="flex items-center gap-2 text-xs">
            <span className={`h-2 w-2 shrink-0 rounded-full ${health === 'healthy' ? 'bg-emerald-400' : health === 'down' ? 'bg-rose-500' : 'bg-amber-400 animate-pulse'}`} />
            <span className="hidden text-slate-400 md:inline">
              API {health === 'healthy' ? 'connected' : health === 'down' ? 'offline' : 'checking…'}
            </span>
          </div>
          <p className="mt-2 hidden text-[10px] leading-snug text-slate-600 md:block">
            Model & metrics are from synthetic demo data — not real-world predictions.
          </p>
        </div>
      </aside>

      {/* main */}
      <main className="thin-scroll min-w-0 flex-1 overflow-y-auto">
        <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-slate-200 bg-white/90 px-8 py-4 backdrop-blur">
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-900">
              {NAV.find((n) => n.id === page)?.label}
            </h1>
            <p className="text-xs text-slate-500">{NAV.find((n) => n.id === page)?.desc}</p>
          </div>
          {simulation && page !== 'simulator' && (
            <button
              onClick={() => setPage('simulator')}
              className="flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-100"
            >
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500" />
              Simulation active: {simulation.scenario_label}
            </button>
          )}
        </header>

        <div className="p-8">
          {health === 'down' && page !== 'overview' ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-8 text-center">
              <p className="text-sm font-semibold text-rose-700">Backend offline</p>
              <p className="mx-auto mt-1 max-w-md text-xs text-rose-600">
                Start it with <code className="rounded bg-white px-1">.venv/Scripts/python -m uvicorn backend.main:app --port 8000</code> then reload.
              </p>
            </div>
          ) : (
            <>
              {page === 'overview' && <OverviewPage />}
              {page === 'graph' && <GraphPage simulation={simulation} />}
              {page === 'predict' && <PredictPage />}
              {page === 'simulator' && (
                <SimulatorPage simulation={simulation} onSimulation={setSimulation} />
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}
