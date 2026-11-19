import { useCallback, useEffect, useState } from 'react'
import { apiFetch } from '../lib/api'

interface MonitorRule {
  id: string
  name: string
  source_table: string
  metric_column: string
  comparison: 'above' | 'below'
  threshold: number
  last_value: number | null
  last_checked_at: string | null
  is_triggered: boolean
  is_active: boolean
}

interface MonitorAlert {
  id: string
  rule_id: string
  rule_name: string
  metric_value: number
  threshold: number
  comparison: 'above' | 'below'
  message: string
  created_at: string
  acknowledged_at: string | null
}

interface MonitorPreview {
  current_value: number
  source_table: string
  metric_column: string
}

const inputClass = 'block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600'

export default function LiveMonitoring() {
  const [name, setName] = useState('')
  const [sourceTable, setSourceTable] = useState('')
  const [metricColumn, setMetricColumn] = useState('')
  const [comparison, setComparison] = useState<'above' | 'below'>('above')
  const [threshold, setThreshold] = useState('')
  const [preview, setPreview] = useState<MonitorPreview | null>(null)
  const [rules, setRules] = useState<MonitorRule[]>([])
  const [alerts, setAlerts] = useState<MonitorAlert[]>([])
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadMonitoringData = useCallback(async () => {
    try {
      const [rulesResponse, alertsResponse] = await Promise.all([
        apiFetch('/monitoring/rules'),
        apiFetch('/monitoring/alerts'),
      ])
      setRules((rulesResponse as { rules: MonitorRule[] }).rules)
      setAlerts((alertsResponse as { alerts: MonitorAlert[] }).alerts)
      setError(null)
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadMonitoringData()
    const timer = window.setInterval(() => void loadMonitoringData(), 30_000)
    return () => window.clearInterval(timer)
  }, [loadMonitoringData])

  const previewMatchesInputs =
    preview?.source_table === sourceTable.trim() && preview?.metric_column === metricColumn.trim()

  async function testSource() {
    setRunning(true)
    setError(null)
    setPreview(null)
    try {
      const response = await apiFetch('/monitoring/preview', {
        method: 'POST',
        body: JSON.stringify({
          source_table: sourceTable.trim(),
          metric_column: metricColumn.trim(),
        }),
      })
      setPreview({
        ...(response as { current_value: number }),
        source_table: sourceTable.trim(),
        metric_column: metricColumn.trim(),
      })
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setRunning(false)
    }
  }

  async function createMonitor(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setRunning(true)
    setError(null)
    try {
      await apiFetch('/monitoring/rules', {
        method: 'POST',
        body: JSON.stringify({
          name: name.trim(),
          source_table: sourceTable.trim(),
          metric_column: metricColumn.trim(),
          comparison,
          threshold: Number(threshold),
        }),
      })
      setName('')
      setThreshold('')
      setPreview(null)
      await loadMonitoringData()
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setRunning(false)
    }
  }

  async function deleteMonitor(ruleId: string) {
    setError(null)
    try {
      await apiFetch(`/monitoring/rules/${encodeURIComponent(ruleId)}`, { method: 'DELETE' })
      await loadMonitoringData()
    } catch (requestError) {
      setError((requestError as Error).message)
    }
  }

  async function acknowledgeAlert(alertId: string) {
    setError(null)
    try {
      await apiFetch(`/monitoring/alerts/${encodeURIComponent(alertId)}/acknowledge`, { method: 'POST' })
      await loadMonitoringData()
    } catch (requestError) {
      setError((requestError as Error).message)
    }
  }

  return (
    <div className="nexus-tool">
      <div className="nexus-tool__intro">
        <h3 className="text-lg font-semibold text-white">Live Monitoring &amp; Alerts</h3>
        <p className="mt-1 text-sm text-slate-400">
          Check a Supabase KPI every 5 minutes and get an in-app alert when it crosses your threshold.
        </p>
        <p className="mt-2 text-xs leading-5 text-slate-500">
          For safety and accurate totals, monitor a public table or view that returns exactly one row for your
          signed-in user, with a <code>user_id</code> column and a numeric KPI column. Create a Postgres view to
          aggregate transaction data. The monitor filters by your user ID and reads only the selected metric.
        </p>
      </div>

      <form onSubmit={createMonitor} className="nexus-tool__form space-y-5">
        <h4 className="font-medium text-white">Create a monitor</h4>
        <label className="block space-y-2 text-sm text-slate-300">
          Monitor name
          <input
            required
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={100}
            placeholder="For example: Revenue threshold"
            className={inputClass}
          />
        </label>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-300">
            Supabase table or view
            <input
              required
              value={sourceTable}
              onChange={(event) => setSourceTable(event.target.value)}
              maxLength={63}
              placeholder="user_kpi_view"
              className={inputClass}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Numeric KPI column
            <input
              required
              value={metricColumn}
              onChange={(event) => setMetricColumn(event.target.value)}
              maxLength={63}
              placeholder="monthly_revenue"
              className={inputClass}
            />
          </label>
        </div>

        <button
          type="button"
          onClick={testSource}
          disabled={running || !sourceTable.trim() || !metricColumn.trim()}
          className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? 'Checking source...' : 'Test source'}
        </button>

        {previewMatchesInputs && (
          <p role="status" className="text-sm text-emerald-300">
            Source connected. Your current value is {preview.current_value}.
          </p>
        )}

        <div className="grid gap-4 sm:grid-cols-3">
          <label className="space-y-2 text-sm text-slate-300">
            Alert when KPI is
            <select
              value={comparison}
              onChange={(event) => setComparison(event.target.value as typeof comparison)}
              className={inputClass}
            >
              <option value="above">At or above</option>
              <option value="below">At or below</option>
            </select>
          </label>
          <label className="space-y-2 text-sm text-slate-300 sm:col-span-2">
            Threshold
            <input
              required
              type="number"
              step="any"
              value={threshold}
              onChange={(event) => setThreshold(event.target.value)}
              placeholder="Enter a numeric threshold"
              className={inputClass}
            />
          </label>
        </div>

        <button
          type="submit"
          disabled={
            running || !previewMatchesInputs || !name.trim() ||
            threshold === '' || !Number.isFinite(Number(threshold))
          }
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? 'Saving monitor...' : 'Create monitor'}
        </button>
      </form>

      {error && <p role="alert" className="rounded-lg border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">{error}</p>}

      <section className="nexus-surface space-y-3 p-5">
        <h4 className="font-medium text-white">Your monitors ({rules.length})</h4>
        {loading && <p className="text-sm text-slate-500">Loading monitors...</p>}
        {!loading && rules.length === 0 && <p className="text-sm text-slate-500">No monitors yet.</p>}
        {rules.map((rule) => (
          <article key={rule.id} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-800 p-3">
            <div>
              <p className="text-sm font-medium text-slate-200">{rule.name}</p>
              <p className="mt-1 text-xs text-slate-500">
                {rule.source_table}.{rule.metric_column} {rule.comparison === 'above' ? '≥' : '≤'} {rule.threshold}
                {' · Current: '}{rule.last_value ?? 'not checked'}
              </p>
              {rule.last_checked_at && (
                <p className="mt-1 text-xs text-slate-600">Checked {new Date(rule.last_checked_at).toLocaleString()}</p>
              )}
            </div>
            <div className="flex items-center gap-3">
              <span className={`rounded-full px-2 py-1 text-xs ${rule.is_triggered ? 'bg-amber-500/10 text-amber-300' : 'bg-emerald-500/10 text-emerald-300'}`}>
                {rule.is_triggered ? 'Threshold crossed' : 'Normal'}
              </span>
              <button
                type="button"
                onClick={() => void deleteMonitor(rule.id)}
                className="text-xs text-slate-400 transition hover:text-red-300"
              >
                Delete
              </button>
            </div>
          </article>
        ))}
      </section>

      <section className="nexus-surface space-y-3 p-5">
        <h4 className="font-medium text-white">In-app alerts ({alerts.length})</h4>
        {alerts.length === 0 && <p className="text-sm text-slate-500">No alerts yet.</p>}
        {alerts.map((alert) => (
          <article key={alert.id} className={`rounded-lg border p-3 ${alert.acknowledged_at ? 'border-slate-800 bg-slate-900/40' : 'border-amber-700/50 bg-amber-950/20'}`}>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-slate-200">{alert.rule_name}</p>
                <p className="mt-1 text-sm text-slate-300">{alert.message}</p>
                <p className="mt-1 text-xs text-slate-500">{new Date(alert.created_at).toLocaleString()}</p>
              </div>
              {!alert.acknowledged_at && (
                <button
                  type="button"
                  onClick={() => void acknowledgeAlert(alert.id)}
                  className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-200 transition hover:bg-slate-800"
                >
                  Acknowledge
                </button>
              )}
            </div>
          </article>
        ))}
      </section>
    </div>
  )
}
