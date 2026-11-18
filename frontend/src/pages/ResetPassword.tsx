import { useState } from 'react'
import type { FormEvent } from 'react'
import { supabase } from '../lib/supabaseClient'

export default function ResetPassword({ onComplete }: { onComplete: () => void }) {
  const [password, setPassword] = useState('')
  const [confirmation, setConfirmation] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [updated, setUpdated] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)

    if (password !== confirmation) {
      setError('The passwords do not match.')
      return
    }

    setLoading(true)
    try {
      const { error: updateError } = await supabase.auth.updateUser({ password })
      if (updateError) {
        setError(updateError.message)
        return
      }
      setUpdated(true)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not update the password.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#080b12] px-4 py-8 text-slate-100 sm:px-6">
      <div className="pointer-events-none absolute -left-40 -top-40 h-[480px] w-[480px] rounded-full bg-blue-600/[0.06] blur-[120px]" />
      <div className="pointer-events-none absolute -bottom-48 -right-32 h-[520px] w-[520px] rounded-full bg-cyan-500/[0.04] blur-[130px]" />

      <section className="relative w-full max-w-lg rounded-3xl border border-white/[0.09] bg-[#0d131d]/95 p-7 shadow-2xl shadow-black/40 sm:p-12">
        <div className="mb-8 flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600">
            <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" aria-hidden="true">
              <path d="m12 3 7.5 4.5v9L12 21l-7.5-4.5v-9L12 3Z" stroke="white" strokeWidth="1.8" />
              <path d="M8.5 9.2 12 7l3.5 2.2v4.3L12 15.7l-3.5-2.2V9.2Z" fill="white" fillOpacity=".9" />
            </svg>
          </span>
          <span className="text-sm font-bold tracking-[0.14em] text-white">NEXUS AI</span>
        </div>

        {updated ? (
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-300">Password updated</p>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight text-white">You're all set.</h1>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Your password has been changed. Continue to your workspace.
            </p>
            <button
              type="button"
              onClick={onComplete}
              className="mt-6 w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-950/25 transition hover:bg-blue-500"
            >
              Continue to workspace
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-300">Account recovery</p>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight text-white">Choose a new password</h1>
              <p className="mt-2 text-sm leading-6 text-slate-400">Use at least 6 characters for your new password.</p>
            </div>

            <label className="block space-y-2 text-xs font-medium text-slate-300">
              New password
              <input
                type="password"
                autoComplete="new-password"
                required
                minLength={6}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="block w-full rounded-xl border border-slate-700/80 bg-[#090e16] px-3.5 py-3 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70 focus:ring-4 focus:ring-cyan-400/10"
              />
            </label>

            <label className="block space-y-2 text-xs font-medium text-slate-300">
              Confirm new password
              <input
                type="password"
                autoComplete="new-password"
                required
                minLength={6}
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                className="block w-full rounded-xl border border-slate-700/80 bg-[#090e16] px-3.5 py-3 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70 focus:ring-4 focus:ring-cyan-400/10"
              />
            </label>

            {error && (
              <p role="alert" className="rounded-xl border border-red-500/20 bg-red-500/[0.06] px-3.5 py-3 text-sm text-red-300">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-950/25 transition hover:bg-blue-500 disabled:cursor-wait disabled:opacity-60"
            >
              {loading ? 'Updating password...' : 'Update password'}
            </button>
          </form>
        )}
      </section>
    </main>
  )
}
