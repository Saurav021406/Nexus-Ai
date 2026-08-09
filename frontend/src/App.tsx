import { useEffect, useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './lib/supabaseClient'
import { apiFetch } from './lib/api'
import Login from './pages/Login'
import UploadDataset, { type WorkspaceTab } from './pages/UploadDataset'

const marketingNavigation = [
  { label: 'Features', href: '#features' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'About', href: '#about' },
  { label: 'Contact us', href: '#contact' },
]

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
      const me = await apiFetch('/me')
      setBackendMessage(`Backend confirmed you are: ${me.email} (id: ${me.id})`)
    } catch (err) {
      setBackendMessage(`Error calling backend: ${(err as Error).message}`)
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
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-400">
        Loading...
      </div>
    )
  }

  if (!session) {
    return <Login onLoggedIn={() => {}} />
  }

  return (
    <div id="home" className="min-h-screen bg-[#070b13] text-slate-100">
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#0b1019]/95 backdrop-blur">
        <div className="mx-auto flex h-[74px] max-w-7xl items-center gap-6 px-5 sm:px-8">
          <a href="#home" className="flex shrink-0 items-center gap-2.5" aria-label="Nexus AI home">
            <span className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-gradient-to-br from-cyan-400 via-blue-500 to-violet-500 shadow-lg shadow-blue-500/20">
              <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5" aria-hidden="true">
                <path d="m12 3 7.5 4.5v9L12 21l-7.5-4.5v-9L12 3Z" stroke="white" strokeWidth="1.8" />
                <path d="M8.5 9.2 12 7l3.5 2.2v4.3L12 15.7l-3.5-2.2V9.2Z" fill="white" fillOpacity=".9" />
              </svg>
            </span>
            <span className="text-sm font-bold tracking-tight text-white">NEXUS AI</span>
          </a>

          <nav className="hidden flex-1 items-center justify-center gap-5 lg:flex" aria-label="Primary navigation">
            <a href="#workspace" className="text-xs font-medium text-cyan-300 transition-colors hover:text-white">
              Data studio
            </a>
            {marketingNavigation.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="text-xs font-medium text-blue-400 transition-colors hover:text-blue-200"
              >
                {item.label}
              </a>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={() => supabase.auth.signOut()}
              className="hidden text-xs font-medium text-blue-300 transition-colors hover:text-white sm:block"
            >
              Sign out
            </button>
            <button
              type="button"
              onClick={() => selectWorkspaceTab('upload')}
              className="rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-blue-500/25 transition hover:brightness-110 sm:px-5"
            >
              Upload dataset
            </button>
          </div>
        </div>
      </header>

      <main id="workspace" className="mx-auto max-w-7xl px-5 py-10 sm:px-8 sm:py-14">
        <div className="mb-8 flex flex-col gap-4 border-b border-slate-800 pb-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">Nexus AI / Data Studio</p>
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Dataset workspace</h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Upload, profile, and analyze your data in dedicated workspace tabs.
            </p>
          </div>
          <p className="text-sm text-slate-500">
            Signed in as <span className="text-slate-300">{session.user.email}</span>
          </p>
        </div>

        <section className="mb-8 flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/50 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-medium text-slate-200">Connection status</p>
            <p className="text-xs text-slate-500">Check that the frontend, backend, and Supabase are connected.</p>
          </div>
          <button
            onClick={checkBackend}
            disabled={checking}
            className="rounded-lg border border-blue-500/40 bg-blue-500/10 px-4 py-2 text-sm font-medium text-blue-300 transition hover:bg-blue-500/20 disabled:opacity-50"
          >
            {checking ? 'Checking...' : 'Check connection'}
          </button>
        </section>

        {backendMessage && (
          <p className="mb-8 rounded-xl border border-slate-800 bg-slate-900 p-3 text-sm text-slate-300">{backendMessage}</p>
        )}

        <UploadDataset
          activeTab={activeWorkspaceTab}
          onTabChange={setActiveWorkspaceTab}
          onDatasetAvailabilityChange={setHasDataset}
        />

        <div className="mt-20 space-y-20 pb-10">
          <section id="features" className="scroll-mt-28">
            <div className="mb-8 max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">Features</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white">Everything needed to understand a dataset faster.</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <FeatureCard
                title="Instant data profiles"
                description="See row and column counts, data types, missing values, unique values, and sample records as soon as a CSV is uploaded."
              />
              <FeatureCard
                title="Domain-aware analysis"
                description="Detect the dataset's likely business domain and run a tailored AI analysis from a dedicated workspace tab."
              />
              <FeatureCard
                title="Focused workspace"
                description="Keep uploads, profiling, and AI findings separate, so your analysis stays clear without a distracting chat overlay."
              />
            </div>
          </section>

          <section id="how-it-works" className="scroll-mt-28 rounded-3xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8">
            <div className="mb-8 max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">How it works</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white">From CSV to clear next steps in three moves.</h2>
            </div>
            <div className="grid gap-6 md:grid-cols-3">
              <Step number="01" title="Upload your CSV" description="Choose a dataset from the Upload Dataset tab." />
              <Step number="02" title="Review the profile" description="Inspect the structure and preview the first rows of your data." />
              <Step number="03" title="Run AI analysis" description="Detect the domain and get focused metrics and recommendations." />
            </div>
          </section>

          <section id="pricing" className="scroll-mt-28">
            <div className="mb-8 max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">Pricing</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white">Flexible access for every stage of your data work.</h2>
              <p className="mt-3 text-sm leading-6 text-slate-400">Individual workspace access and team deployments can be configured around your data volume and support needs.</p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <PlanCard title="Workspace" description="For focused dataset exploration." features={['CSV upload and data profiles', 'AI domain detection', 'Dedicated analysis tabs']} />
              <PlanCard title="Team deployment" description="For organizations that need a tailored rollout." features={['Custom workspace setup', 'Team onboarding support', 'Deployment and access planning']} featured />
            </div>
          </section>

          <section id="about" className="scroll-mt-28 grid gap-8 border-y border-slate-800 py-12 md:grid-cols-[1.1fr_.9fr] md:items-center">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">About Nexus AI</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white">Data analysis that stays simple and actionable.</h2>
            </div>
            <p className="text-sm leading-7 text-slate-400">
              Nexus AI turns an uploaded CSV into a clear profile and domain-specific findings, helping teams understand their data before they make decisions.
            </p>
          </section>

          <section id="contact" className="scroll-mt-28 rounded-3xl bg-gradient-to-r from-blue-600 to-indigo-600 p-7 sm:p-10">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-blue-100">Contact us</p>
            <div className="mt-3 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-3xl font-semibold tracking-tight text-white">Need help setting up your workspace?</h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-blue-100">Reach out through your organization’s Nexus AI support channel for access, onboarding, or deployment questions.</p>
              </div>
              <a href="#workspace" className="shrink-0 rounded-full bg-white px-5 py-3 text-center text-sm font-semibold text-blue-700 transition hover:bg-blue-50">
                Open data studio
              </a>
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <article className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5 transition hover:-translate-y-1 hover:border-blue-500/50">
      <span className="mb-5 flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-sm font-semibold text-cyan-300">✦</span>
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
    </article>
  )
}

function Step({ number, title, description }: { number: string; title: string; description: string }) {
  return (
    <article>
      <p className="text-sm font-semibold text-cyan-300">{number}</p>
      <h3 className="mt-3 text-lg font-semibold text-white">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
    </article>
  )
}

function PlanCard({
  title,
  description,
  features,
  featured = false,
}: {
  title: string
  description: string
  features: string[]
  featured?: boolean
}) {
  return (
    <article className={`rounded-2xl border p-6 ${featured ? 'border-blue-500/60 bg-blue-500/10' : 'border-slate-800 bg-slate-900/40'}`}>
      <h3 className="text-xl font-semibold text-white">{title}</h3>
      <p className="mt-2 text-sm text-slate-400">{description}</p>
      <p className="mt-6 text-sm font-semibold text-cyan-300">Custom pricing</p>
      <ul className="mt-5 space-y-3">
        {features.map((feature) => (
          <li key={feature} className="flex gap-2 text-sm text-slate-300">
            <span className="text-cyan-300">✓</span>
            {feature}
          </li>
        ))}
      </ul>
    </article>
  )
}
