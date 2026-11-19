import { useState } from 'react'
import { apiFetch } from '../lib/api'

interface MeetingTask {
  task: string
  owner: string | null
  deadline: string | null
  priority: 'high' | 'medium' | 'low'
}

interface MeetingRisk {
  risk: string
  impact: string
  mitigation: string | null
}

interface MeetingResult {
  summary: string
  tasks: MeetingTask[]
  decisions: string[]
  risks: MeetingRisk[]
  open_questions: string[]
}

export default function MeetingIntelligence() {
  const [transcript, setTranscript] = useState('')
  const [consent, setConsent] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<MeetingResult | null>(null)

  async function analyzeTranscript(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const response = await apiFetch('/meeting-intelligence/analyze', {
        method: 'POST',
        body: JSON.stringify({ transcript: transcript.trim(), consent_to_ai_processing: consent }),
      })
      setResult(response as MeetingResult)
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="nexus-tool">
      <div className="nexus-tool__intro">
        <h3 className="text-lg font-semibold text-white">Meeting Intelligence</h3>
        <p className="mt-1 text-sm text-slate-400">
          Turn a transcript into a concise summary, action items, decisions, risks, and open questions.
        </p>
      </div>

      <form onSubmit={analyzeTranscript} className="nexus-tool__form space-y-5">
        <label className="block space-y-2 text-sm text-slate-300">
          Meeting transcript
          <textarea
            required
            value={transcript}
            onChange={(event) => setTranscript(event.target.value)}
            maxLength={30000}
            rows={12}
            placeholder="Paste the meeting transcript here..."
            className="block w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600"
          />
        </label>
        <p className="text-xs text-slate-500">{transcript.length.toLocaleString()} / 30,000 characters</p>
        <label className="flex items-start gap-3 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
            className="mt-1 accent-blue-500"
          />
          <span>
            I have permission to share this transcript with the AI provider configured for this workspace.
            Transcript text is sent for analysis.
          </span>
        </label>
        <button
          type="submit"
          disabled={running || !transcript.trim() || !consent}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? 'Analyzing meeting...' : 'Analyze meeting'}
        </button>
      </form>

      {error && <p role="alert" className="rounded-lg border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">{error}</p>}

      {result && (
        <div className="space-y-4">
          <section className="nexus-surface p-5">
            <h4 className="font-medium text-white">Summary</h4>
            <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-300">{result.summary}</p>
          </section>

          <section className="nexus-surface p-5">
            <h4 className="font-medium text-white">Action items ({result.tasks.length})</h4>
            {result.tasks.length ? (
              <ul className="mt-3 space-y-3">
                {result.tasks.map((task, index) => (
                  <li key={`${task.task}-${index}`} className="rounded-lg border border-slate-800 p-3">
                    <p className="text-sm text-slate-200">{task.task}</p>
                    <p className="mt-1 text-xs text-slate-500">
                      Owner: {task.owner || 'Not stated'} · Due: {task.deadline || 'Not stated'} · Priority: {task.priority}
                    </p>
                  </li>
                ))}
              </ul>
            ) : <p className="mt-2 text-sm text-slate-500">No action items identified.</p>}
          </section>

          <div className="grid gap-4 md:grid-cols-2">
            <ListSection title="Decisions" items={result.decisions} />
            <ListSection title="Open questions" items={result.open_questions} />
          </div>

          <section className="nexus-surface p-5">
            <h4 className="font-medium text-white">Risks ({result.risks.length})</h4>
            {result.risks.length ? (
              <ul className="mt-3 space-y-3">
                {result.risks.map((risk, index) => (
                  <li key={`${risk.risk}-${index}`} className="rounded-lg border border-slate-800 p-3">
                    <p className="text-sm text-slate-200">{risk.risk}</p>
                    <p className="mt-1 text-xs text-slate-400">Impact: {risk.impact}</p>
                    {risk.mitigation && <p className="mt-1 text-xs text-slate-500">Mitigation: {risk.mitigation}</p>}
                  </li>
                ))}
              </ul>
            ) : <p className="mt-2 text-sm text-slate-500">No risks identified.</p>}
          </section>
        </div>
      )}
    </div>
  )
}

function ListSection({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="nexus-surface p-5">
      <h4 className="font-medium text-white">{title}</h4>
      {items.length ? (
        <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-slate-300">
          {items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
        </ul>
      ) : <p className="mt-2 text-sm text-slate-500">None identified.</p>}
    </section>
  )
}
