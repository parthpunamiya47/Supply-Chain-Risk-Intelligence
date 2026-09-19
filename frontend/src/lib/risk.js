// Shared risk-band helpers - the single source of UI truth for risk colors.
// Bands mirror the backend: LOW < 40, MEDIUM 40-69, HIGH >= 70 (percent scale).

export const RISK_COLORS = {
  LOW: {
    text: 'text-emerald-700',
    bg: 'bg-emerald-500',
    bgSoft: 'bg-emerald-50',
    border: 'border-emerald-300',
    hex: '#10b981',
    ring: 'ring-emerald-200',
  },
  MEDIUM: {
    text: 'text-amber-700',
    bg: 'bg-amber-500',
    bgSoft: 'bg-amber-50',
    border: 'border-amber-300',
    hex: '#f59e0b',
    ring: 'ring-amber-200',
  },
  HIGH: {
    text: 'text-rose-700',
    bg: 'bg-rose-600',
    bgSoft: 'bg-rose-50',
    border: 'border-rose-300',
    hex: '#e11d48',
    ring: 'ring-rose-200',
  },
}

export const band = (riskPercent) => {
  if (riskPercent == null || Number.isNaN(riskPercent)) return 'LOW'
  if (riskPercent >= 70) return 'HIGH'
  if (riskPercent >= 40) return 'MEDIUM'
  return 'LOW'
}

export const bandOf01 = (risk01) => band((risk01 ?? 0) * 100)

export const colorFor = (riskPercent) => RISK_COLORS[band(riskPercent)]

export const hexFor01 = (risk01) => RISK_COLORS[bandOf01(risk01)].hex

export const PRIORITY_STYLES = {
  HIGH: 'bg-rose-100 text-rose-700 border-rose-200',
  MEDIUM: 'bg-amber-100 text-amber-700 border-amber-200',
  LOW: 'bg-sky-100 text-sky-700 border-sky-200',
  NONE: 'bg-slate-100 text-slate-500 border-slate-200',
}

export const fmtPct = (v, digits = 1) =>
  v == null || Number.isNaN(Number(v)) ? '—' : `${Number(v).toFixed(digits)}%`

export const fmtNum = (v) =>
  v == null ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: 1 })
