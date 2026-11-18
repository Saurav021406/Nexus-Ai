import { useEffect, useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './lib/supabaseClient'
import { apiFetch } from './lib/api'
import Login from './pages/Login'
import UploadDataset, { type WorkspaceTab } from './pages/UploadDataset'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [loadingSession, setLoadingSession] = useState(true)
  const [backendMessage, setBackendMessage] = useState<string>('')
  const [checking, setChecking] = useState(false)
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState<WorkspaceTab>('upload')
  const [hasDataset, setHasDataset] = useState(false)

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setLoadingSession(false)
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession)
    })

    return () => listener.subscription.unsubscribe()
  }, [])

  async function checkBackend() {
    setChecking(true)
    setBackendMessage('')
    try {
      await apiFetch('/me')
      setBackendMessage('Backend and authentication are connected.')
    } catch (err) {
      setBackendMessage(`Connection check failed: ${(err as Error).message}`)
    } finally {
      setChecking(false)
    }
  }

  function selectWorkspaceTab(tab: WorkspaceTab) {
    if (tab !== 'upload' && !hasDataset) return
    setActiveWorkspaceTab(tab)
    document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  if (loadingSession) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#080b12] text-sm text-slate-400">
        <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/70 px-5 py-4">
          <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
          Loading your workspace...
        </div>
      </div>
    )
  }

  if (!session) {
    return <Login onLoggedIn={() => {}} />
  }

  return (
    <div id="home" className="min-h-screen bg-[#080b12] text-slate-100">
      <header className="sticky top-0 z-40 border-b border-white/[0.07] bg-[#080b12]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[68px] max-w-[1600px] items-center justify-between gap-4 px-4 sm:px-7">
          <a href="#home" className="flex shrink-0 items-center gap-3" aria-label="Nexus AI workspace home">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 shadow-lg shadow-blue-900/30">
              <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" aria-hidden="true">
                <path d="m12 3 7.5 4.5v9L12 21l-7.5-4.5v-9L12 3Z" stroke="white" strokeWidth="1.8" />
                <path d="M8.5 9.2 12 7l3.5 2.2v4.3L12 15.7l-3.5-2.2V9.2Z" fill="white" fillOpacity=".9" />
              </svg>
            </span>
            <span>
              <span className="block text-sm font-bold tracking-[0.12em] text-white">NEXUS AI</span>
              <span className="mt-0.5 hidden text-[11px] text-slate-500 sm:block">DATA WORKSPACE</span>
            </span>
          </a>

          <div className="flex min-w-0 items-center gap-3">
            <div className="hidden min-w-0 items-center gap-2.5 sm:flex">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-800 text-xs font-semibold text-slate-200">
                {(session.user.email?.[0] ?? 'U').toUpperCase()}
              </span>
              <span className="max-w-[220px] truncate text-xs text-slate-400">{session.user.email}</span>
            </div>
            <button
              onClick={() => void supabase.auth.signOut()}
              className="rounded-lg border border-slate-700/80 px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-slate-600 hover:bg-white/[0.04] hover:text-white"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main id="workspace" className="mx-auto max-w-[1600px] px-4 pb-12 pt-8 sm:px-7 sm:pt-10">
        <div className="mb-7 flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-400/15 bg-cyan-400/[0.06] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-300">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
              Nexus AI workspace
            </p>
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Your data, ready to work.</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Profile datasets, run focused AI workflows, and keep your workspace activity in one place.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.025] px-4 py-3">
              <span className={`h-2 w-2 rounded-full ${hasDataset ? 'bg-emerald-400' : 'bg-slate-500'}`} />
              <div>
                <p className="text-xs font-medium text-slate-200">{hasDataset ? 'Dataset ready' : 'No dataset loaded'}</p>
                <p className="mt-0.5 text-[11px] text-slate-500">{hasDataset ? 'Analysis tools available' : 'Upload to unlock data tools'}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={checkBackend}
                disabled={checking}
                className="rounded-xl border border-slate-700/80 bg-slate-900/60 px-4 py-3 text-xs font-semibold text-slate-200 transition hover:border-slate-600 hover:bg-slate-800/70 disabled:cursor-wait disabled:opacity-60"
              >
                {checking ? 'Checking...' : 'Check connection'}
              </button>
              <button
                type="button"
                onClick={() => selectWorkspaceTab('upload')}
                className="rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-4 py-3 text-xs font-semibold text-white shadow-lg shadow-blue-950/30 transition hover:brightness-110"
              >
                Upload data
              </button>
            </div>
          </div>
        </div>

        {backendMessage && (
          <p
            role={backendMessage.startsWith('Connection check failed:') ? 'alert' : 'status'}
            className={`mb-5 rounded-xl border px-4 py-3 text-sm ${
              backendMessage.startsWith('Connection check failed:')
                ? 'border-red-500/20 bg-red-500/[0.06] text-red-300'
                : 'border-emerald-500/20 bg-emerald-500/[0.06] text-emerald-300'
            }`}
          >
            {backendMessage}
          </p>
        )}

        <UploadDataset
          activeTab={activeWorkspaceTab}
          onTabChange={setActiveWorkspaceTab}
          onDatasetAvailabilityChange={setHasDataset}
        />
      </main>
    </div>
  )
}
