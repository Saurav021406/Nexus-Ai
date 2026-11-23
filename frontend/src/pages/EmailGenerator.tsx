import { useState } from 'react'
import type { FormEvent } from 'react'
import { apiFetch } from '../lib/api'

interface EmailDraft {
  subject: string
  body: string
}

const emailTemplates = [
  { value: 'status', label: 'Project status update', purpose: 'Request a project status update' },
  { value: 'meeting', label: 'Meeting follow-up', purpose: 'Follow up on a meeting with key points and next steps' },
  { value: 'request', label: 'Make a request', purpose: 'Make a clear and polite request' },
  { value: 'thanks', label: 'Thank you', purpose: 'Write a thank-you email' },
  { value: 'apology', label: 'Apology', purpose: 'Apologize for an issue and explain the next steps' },
  { value: 'introduction', label: 'Introduction', purpose: 'Introduce yourself and explain why you are reaching out' },
]

export default function EmailGenerator() {
  const [template, setTemplate] = useState('')
  const [purpose, setPurpose] = useState('')
  const [recipient, setRecipient] = useState('')
  const [context, setContext] = useState('')
  const [signature, setSignature] = useState('')
  const [tone, setTone] = useState<'formal' | 'warm' | 'friendly' | 'persuasive' | 'apologetic'>('formal')
  const [length, setLength] = useState<'concise' | 'standard' | 'detailed'>('standard')
  const [consent, setConsent] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copyMessage, setCopyMessage] = useState('')
  const [draft, setDraft] = useState<EmailDraft | null>(null)

  async function generateDraft(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setRunning(true)
    setError(null)
    setCopyMessage('')
    setDraft(null)
    try {
      const response = await apiFetch('/email-generator/generate', {
        method: 'POST',
        body: JSON.stringify({
          purpose: purpose.trim(),
          context: context.trim(),
          recipient: recipient.trim() || null,
          signature: signature || null,
          tone,
          length,
          consent_to_ai_processing: consent,
        }),
      })
      setDraft(response as EmailDraft)
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setRunning(false)
    }
  }

  async function copyText(text: string, label: string) {
    try {
      await navigator.clipboard.writeText(text)
      setCopyMessage(`${label} copied to clipboard.`)
      setError(null)
    } catch (copyError) {
      setError(`Could not copy the draft: ${(copyError as Error).message}`)
      setCopyMessage('')
    }
  }

  return (
    <div className="nexus-tool">
      <div className="nexus-tool__intro">
        <h3 className="text-lg font-semibold text-white">Email Generator</h3>
        <p className="mt-1 text-sm text-slate-400">
          Create an editable email draft from your goal and key details. Review it before sending; this tool does not send emails.
        </p>
      </div>

      <form onSubmit={generateDraft} className="nexus-tool__form space-y-5">
        <label className="block space-y-2 text-sm text-slate-300">
          Start from a template
          <select
            value={template}
            onChange={(event) => {
              const selectedTemplate = emailTemplates.find((item) => item.value === event.target.value)
              setTemplate(event.target.value)
              if (selectedTemplate) setPurpose(selectedTemplate.purpose)
            }}
            className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
          >
            <option value="">Choose a template (optional)</option>
            {emailTemplates.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </label>

        <label className="block space-y-2 text-sm text-slate-300">
          What should the email accomplish?
          <input
            required
            value={purpose}
            onChange={(event) => {
              setPurpose(event.target.value)
              setTemplate('')
            }}
            maxLength={1000}
            placeholder="For example: Request a project status update"
            className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder:text-slate-600"
          />
        </label>

        <label className="block space-y-2 text-sm text-slate-300">
          Key details to include
          <textarea
            required
            value={context}
            onChange={(event) => setContext(event.target.value)}
            maxLength={10000}
            rows={5}
            placeholder="Add relevant facts, dates, requests, or next steps. The draft will not invent missing details."
            className="block w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600"
          />
          <span className="block text-xs text-slate-500">{context.length.toLocaleString()} / 10,000 characters</span>
        </label>

        <div className="grid gap-4 sm:grid-cols-3">
          <label className="space-y-2 text-sm text-slate-300">
            Recipient <span className="text-slate-500">(optional)</span>
            <input
              value={recipient}
              onChange={(event) => setRecipient(event.target.value)}
              maxLength={200}
              placeholder="Name or audience"
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder:text-slate-600"
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Tone
            <select
              value={tone}
              onChange={(event) => setTone(event.target.value as typeof tone)}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
            >
              <option value="formal">Formal</option>
              <option value="warm">Warm</option>
              <option value="friendly">Friendly</option>
              <option value="persuasive">Persuasive</option>
              <option value="apologetic">Apologetic</option>
            </select>
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Length
            <select
              value={length}
              onChange={(event) => setLength(event.target.value as typeof length)}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
            >
              <option value="concise">Concise</option>
              <option value="standard">Standard</option>
              <option value="detailed">Detailed</option>
            </select>
          </label>
        </div>

        <label className="block space-y-2 text-sm text-slate-300">
          Signature <span className="text-slate-500">(optional; appended exactly as written)</span>
          <textarea
            value={signature}
            onChange={(event) => setSignature(event.target.value)}
            maxLength={1000}
            rows={3}
            placeholder={'Best,\nYour name'}
            className="block w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600"
          />
          <span className="block text-xs text-slate-500">Appended in this workspace; your signature is not sent to the AI provider.</span>
        </label>

        <label className="flex items-start gap-3 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
            className="mt-1 accent-blue-500"
          />
          <span>
            I have permission to share these details with the AI provider configured for this workspace. The text is sent for draft generation.
          </span>
        </label>

        <button
          type="submit"
          disabled={running || !purpose.trim() || !context.trim() || !consent}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? 'Generating draft...' : 'Generate email draft'}
        </button>
      </form>

      {error && <p role="alert" className="rounded-lg border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">{error}</p>}

      {draft && (
        <section className="nexus-surface space-y-4 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h4 className="font-medium text-white">Review and edit your draft</h4>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => void copyText(`Subject: ${draft.subject}\n\n${draft.body}`, 'Draft')}
                className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
              >
                Copy draft
              </button>
              <button
                type="button"
                onClick={() => void copyText(draft.subject, 'Subject')}
                className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
              >
                Copy subject
              </button>
              <button
                type="button"
                onClick={() => void copyText(draft.body, 'Body')}
                className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
              >
                Copy body
              </button>
            </div>
          </div>
          {copyMessage && <p role="status" className="text-sm text-emerald-300">{copyMessage}</p>}
          <label className="block space-y-2 text-sm text-slate-300">
            Subject
            <input
              value={draft.subject}
              onChange={(event) => setDraft({ ...draft, subject: event.target.value })}
              maxLength={250}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
            />
          </label>
          <label className="block space-y-2 text-sm text-slate-300">
            Email body
            <textarea
              value={draft.body}
              onChange={(event) => setDraft({ ...draft, body: event.target.value })}
              maxLength={12000}
              rows={12}
              className="block w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm leading-6 text-slate-100"
            />
          </label>
        </section>
      )}
    </div>
  )
}
