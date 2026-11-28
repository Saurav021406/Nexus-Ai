import { useCallback, useEffect, useState } from 'react'
import { apiFetch } from '../lib/api'

interface AuditEvent {
  id: string
  action: string
  endpoint: string
  status_code: number
  duration_ms: number
  created_at: string
}

export default function AuditLogs() {
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadEvents = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiFetch('/audit-logs?limit=100')
      setEvents((response as { events: AuditEvent[] }).events)
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadEvents()
  }, [loadEvents])

  return (
    <div className="nexus-tool">
      <div className="nexus-tool__intro flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-white">Audit Logs</h3>
          <p className="mt-1 text-sm text-slate-400">
            Recent authenticated actions, including AI requests and changes. Request and response content is not stored.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void loadEvents()}
          disabled={loading}
          className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-200 transition hover:bg-slate-800 disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && <p role="alert" className="rounded-lg border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">{error}</p>}

      <section className="nexus-surface overflow-hidden">
        {loading && events.length === 0 && <p className="p-4 text-sm text-slate-500">Loading audit history...</p>}
        {!loading && events.length === 0 && !error && <p className="p-4 text-sm text-slate-500">No actions recorded yet.</p>}
        {events.length > 0 && (
          <div className="divide-y divide-slate-800">
            {events.map((event) => (
              <article key={event.id} className="flex flex-wrap items-center justify-between gap-3 p-4 transition hover:bg-white/[0.02]">
                <div className="min-w-0">
                  <p className="break-all text-sm font-medium text-slate-200">{event.action}</p>
                  <p className="mt-1 text-xs text-slate-500">{new Date(event.created_at).toLocaleString()}</p>
                </div>
                <div className="flex shrink-0 items-center gap-3 text-xs">
                  <span className={`rounded-full px-2 py-1 ${event.status_code < 400 ? 'bg-emerald-500/10 text-emerald-300' : 'bg-red-500/10 text-red-300'}`}>
                    {event.status_code}
                  </span>
                  <span className="text-slate-500">{event.duration_ms} ms</span>
                </div>
              </article>
            ))}
          </div>
        )}
        {events.length > 0 && <p className="border-t border-slate-800 p-3 text-xs text-slate-500">Showing the latest {events.length} event(s).</p>}
      </section>
    </div>
  )
}
