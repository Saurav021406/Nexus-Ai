import { useState } from 'react'
import { supabase } from '../lib/supabaseClient'

export default function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [mode, setMode] = useState<'sign-in' | 'sign-up' | 'forgot-password'>('sign-in')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setNotice(null)
    setLoading(true)

    try {
      if (mode === 'forgot-password') {
        const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/?mode=reset-password`,
        })
        if (resetError) throw resetError
        setNotice('If an account exists for that email, a password reset link is on its way.')
        return
      }

      const { error: authError } = mode === 'sign-in'
        ? await supabase.auth.signInWithPassword({ email, password })
        : await supabase.auth.signUp({ email, password })

      if (authError) throw authError
      onLoggedIn()
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Authentication request failed.')
    } finally {
      setLoading(false)
    }
  }

  const isSignIn = mode === 'sign-in'
  const isForgotPassword = mode === 'forgot-password'

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#080b12] px-4 py-8 text-slate-100 sm:px-6">
      <div className="pointer-events-none absolute -left-40 -top-40 h-[480px] w-[480px] rounded-full bg-blue-600/[0.06] blur-[120px]" />
      <div className="pointer-events-none absolute -bottom-48 -right-32 h-[520px] w-[520px] rounded-full bg-cyan-500/[0.04] blur-[130px]" />

      <section className="relative grid w-full max-w-6xl overflow-hidden rounded-3xl border border-white/[0.09] bg-[#0d131d]/95 shadow-2xl shadow-black/40 lg:grid-cols-[1.05fr_.95fr]">
        <div className="hidden flex-col justify-between border-r border-white/[0.07] bg-gradient-to-br from-blue-500/[0.06] via-transparent to-cyan-400/[0.025] p-12 lg:flex xl:p-14">
          <a href="/" className="inline-flex w-fit items-center gap-3" aria-label="Nexus AI home">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 shadow-lg shadow-blue-950/40">
              <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" aria-hidden="true">
                <path d="m12 3 7.5 4.5v9L12 21l-7.5-4.5v-9L12 3Z" stroke="white" strokeWidth="1.8" />
                <path d="M8.5 9.2 12 7l3.5 2.2v4.3L12 15.7l-3.5-2.2V9.2Z" fill="white" fillOpacity=".9" />
              </svg>
            </span>
            <span className="text-sm font-bold tracking-[0.14em] text-white">NEXUS AI</span>
          </a>

          <div className="my-12">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-300">A clearer way to work with data</p>
            <h1 className="mt-4 max-w-lg text-4xl font-semibold leading-tight tracking-tight text-white xl:text-5xl">
              Turn complex data into confident next steps.
            </h1>
            <p className="mt-5 max-w-md text-sm leading-7 text-slate-400">
              Explore datasets, run focused AI workflows, and keep important workspace activity together.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
            {[
              ['01', 'Explore data'],
              ['02', 'Run AI tools'],
              ['03', 'Review activity'],
            ].map(([number, label]) => (
              <div key={number} className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-3">
                <span className="text-[11px] font-semibold text-cyan-300">{number}</span>
                <span className="text-xs font-medium text-slate-300">{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center p-6 sm:p-12 xl:p-14">
          <form onSubmit={handleSubmit} className="mx-auto w-full max-w-md space-y-7">
            <div className="lg:hidden">
              <a href="/" className="mb-8 inline-flex items-center gap-2.5" aria-label="Nexus AI home">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600">
                  <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" aria-hidden="true">
                    <path d="m12 3 7.5 4.5v9L12 21l-7.5-4.5v-9L12 3Z" stroke="white" strokeWidth="1.8" />
                    <path d="M8.5 9.2 12 7l3.5 2.2v4.3L12 15.7l-3.5-2.2V9.2Z" fill="white" fillOpacity=".9" />
                  </svg>
                </span>
                <span className="text-sm font-bold tracking-[0.14em] text-white">NEXUS AI</span>
              </a>
            </div>

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-300">
                {isForgotPassword ? 'Account recovery' : isSignIn ? 'Welcome back' : 'Get started'}
              </p>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white">
                {isForgotPassword ? 'Reset your password' : isSignIn ? 'Sign in to your workspace' : 'Create your account'}
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                {isForgotPassword
                  ? 'Enter the email address linked to your account and we will send a reset link.'
                  : isSignIn
                    ? 'Enter your details to continue to Nexus AI.'
                    : 'Create an account to set up your Nexus AI workspace.'}
              </p>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="login-email" className="block text-xs font-medium text-slate-300">Email address</label>
                <input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="block w-full rounded-xl border border-slate-700/80 bg-[#090e16] px-3.5 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-400/70 focus:ring-4 focus:ring-cyan-400/10"
                />
              </div>

              {!isForgotPassword && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <label htmlFor="login-password" className="block text-xs font-medium text-slate-300">Password</label>
                    {isSignIn && (
                      <button
                        type="button"
                        onClick={() => {
                          setError(null)
                          setNotice(null)
                          setMode('forgot-password')
                        }}
                        className="text-xs font-medium text-cyan-300 transition hover:text-cyan-200"
                      >
                        Forgot password?
                      </button>
                    )}
                  </div>
                  <input
                    id="login-password"
                    type="password"
                    autoComplete={isSignIn ? 'current-password' : 'new-password'}
                    required
                    minLength={6}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="At least 6 characters"
                    className="block w-full rounded-xl border border-slate-700/80 bg-[#090e16] px-3.5 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-400/70 focus:ring-4 focus:ring-cyan-400/10"
                  />
                </div>
              )}
            </div>

            {error && (
              <p role="alert" className="rounded-xl border border-red-500/20 bg-red-500/[0.06] px-3.5 py-3 text-sm text-red-300">
                {error}
              </p>
            )}
            {notice && (
              <p role="status" className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] px-3.5 py-3 text-sm text-emerald-300">
                {notice}
              </p>
            )}

            <div className="space-y-4">
              <div className="space-y-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-950/25 transition hover:bg-blue-500 disabled:cursor-wait disabled:opacity-60"
                >
                  {loading
                    ? 'Please wait...'
                    : isForgotPassword
                      ? 'Send reset link'
                      : isSignIn
                        ? 'Sign in'
                        : 'Create account'}
                </button>
                {isForgotPassword && (
                  <button
                    type="button"
                    onClick={() => {
                      setError(null)
                      setNotice(null)
                      setMode('sign-in')
                    }}
                    className="w-full rounded-xl border border-slate-700/80 px-4 py-3 text-sm font-medium text-slate-300 transition hover:bg-white/[0.04] hover:text-white"
                  >
                    Back to sign in
                  </button>
                )}
              </div>

              {!isForgotPassword && (
                <p className="text-center text-sm text-slate-500">
                  {isSignIn ? "Don't have an account?" : 'Already have an account?'}{' '}
                  <button
                    type="button"
                    onClick={() => {
                      setError(null)
                      setNotice(null)
                      setMode(isSignIn ? 'sign-up' : 'sign-in')
                    }}
                    className="font-medium text-cyan-300 transition hover:text-cyan-200"
                  >
                    {isSignIn ? 'Create account' : 'Sign in'}
                  </button>
                </p>
              )}
            </div>
          </form>
        </div>
      </section>
    </main>
  )
}
