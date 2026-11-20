import { useState } from 'react'
import { supabase } from '../lib/supabaseClient'

export default function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [mode, setMode] = useState<'sign-in' | 'sign-up' | 'forgot-password'>('sign-in')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [oauthLoading, setOauthLoading] = useState<'google' | 'github' | null>(null)

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

  async function handleOAuthSignIn(provider: 'google' | 'github') {
    setError(null)
    setNotice(null)
    setOauthLoading(provider)

    try {
      const { error: oauthError } = await supabase.auth.signInWithOAuth({
        provider,
        options: { redirectTo: window.location.origin },
      })
      if (oauthError) throw oauthError
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Social sign-in failed.')
      setOauthLoading(null)
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

            {!isForgotPassword && (
              <div className="space-y-3">
                <button
                  type="button"
                  disabled={loading || oauthLoading !== null}
                  onClick={() => void handleOAuthSignIn('google')}
                  className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-700/80 bg-white px-4 py-3 text-sm font-semibold text-slate-800 transition hover:bg-slate-100 disabled:cursor-wait disabled:opacity-60"
                >
                  <svg viewBox="0 0 48 48" className="h-5 w-5" aria-hidden="true">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5Z" transform="translate(0 5)" />
                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.75 7.18l7.73 6C44.43 38.12 46.98 31.95 46.98 24.55Z" />
                    <path fill="#FBBC05" d="M10.53 28.59a14.4 14.4 0 0 1 0-9.18l-7.98-6.19a23.9 23.9 0 0 0 0 21.56l7.98-6.19Z" transform="translate(0 5)" />
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.91-5.8l-7.73-6c-2.14 1.44-4.88 2.3-8.18 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48Z" transform="translate(0 -5)" />
                  </svg>
                  {oauthLoading === 'google' ? 'Connecting to Google...' : 'Continue with Google'}
                </button>
                <button
                  type="button"
                  disabled={loading || oauthLoading !== null}
                  onClick={() => void handleOAuthSignIn('github')}
                  className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-700/80 bg-[#161b22] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#22272e] disabled:cursor-wait disabled:opacity-60"
                >
                  <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
                    <path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.56.1.76-.24.76-.54v-2.08c-3.1.67-3.76-1.32-3.76-1.32-.5-1.28-1.24-1.62-1.24-1.62-1.02-.7.08-.69.08-.69 1.13.08 1.73 1.16 1.73 1.16 1 .1.76 2.3 3.87 1.63.1-.73.4-1.23.7-1.52-2.48-.28-5.09-1.24-5.09-5.52 0-1.22.44-2.22 1.16-3-.12-.29-.5-1.43.11-2.98 0 0 .95-.3 3.05 1.15a10.6 10.6 0 0 1 5.55 0c2.1-1.43 3.05-1.15 3.05-1.15.61 1.55.23 2.7.11 2.98.72.78 1.16 1.78 1.16 3 0 4.29-2.61 5.23-5.1 5.51.4.35.76 1.03.76 2.08v3.08c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z" />
                  </svg>
                  {oauthLoading === 'github' ? 'Connecting to GitHub...' : 'Continue with GitHub'}
                </button>
              </div>
            )}

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
