import { RISK_COLORS, PRIORITY_STYLES, band } from '../lib/risk'

export function Card({ title, subtitle, children, className = '', bodyClassName = '', actions = null }) {
  return (
    <section
      className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-slate-100 px-5 py-3.5">
          <div>
            {title && (
              <h2 className="text-sm font-semibold tracking-tight text-slate-800">{title}</h2>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={`p-5 ${bodyClassName}`}>{children}</div>
    </section>
  )
}

export function KpiCard({ label, value, sub, tone = 'slate', loading }) {
  const tones = {
    slate: 'text-slate-900',
    emerald: 'text-emerald-600',
    amber: 'text-amber-600',
    rose: 'text-rose-600',
    sky: 'text-sky-600',
  }
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
      {loading ? (
        <div className="mt-2 h-8 w-16 animate-pulse rounded bg-slate-200" />
      ) : (
        <p className={`num mt-1 text-3xl font-semibold tracking-tight ${tones[tone]}`}>{value}</p>
      )}
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  )
}

export function Badge({ children, tone = 'slate', className = '' }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-600 border-slate-200',
    ...RISK_COLORS,
    ...PRIORITY_STYLES,
  }
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${tones[tone] || tones.slate} ${className}`}
    >
      {children}
    </span>
  )
}

export function RiskBar({ percent }) {
  const c = RISK_COLORS[band(percent)]
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
      <div
        className={`h-full rounded-full ${c.bg}`}
        style={{ width: `${Math.min(100, Math.max(2, percent))}%` }}
      />
    </div>
  )
}

export function SectionTitle({ children, hint }) {
  return (
    <div className="mb-3 flex items-baseline justify-between">
      <h3 className="text-sm font-semibold text-slate-800">{children}</h3>
      {hint && <span className="text-xs text-slate-400">{hint}</span>}
    </div>
  )
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-rose-300 bg-rose-50 p-6 text-center">
      <p className="text-sm font-medium text-rose-700">Backend unreachable</p>
      <p className="max-w-md text-xs text-rose-600">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 rounded-md border border-rose-300 bg-white px-3 py-1 text-xs font-medium text-rose-700 hover:bg-rose-100"
        >
          Retry
        </button>
      )}
    </div>
  )
}

export function LoadingState({ label = 'Loading…' }) {
  return (
    <div className="flex h-full min-h-40 items-center justify-center gap-2 text-sm text-slate-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
      {label}
    </div>
  )
}

export function EmptyState({ children }) {
  return (
    <div className="flex min-h-40 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-500">
      {children}
    </div>
  )
}
