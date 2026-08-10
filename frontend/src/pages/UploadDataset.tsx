import { useEffect, useRef, useState } from 'react'
import { supabase } from '../lib/supabaseClient'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string

interface ColumnInfo {
  name: string
  dtype: string
  missing_count: number
  missing_pct: number
  unique_count: number
  stats?: { mean: number; min: number; max: number; std: number }
}

interface AnalysisResult {
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  columns: ColumnInfo[]
  preview_rows: Record<string, string>[]
}

interface DomainResult {
  primary_domain: string
  secondary_domains: string[]
  tags: string[]
  confidence: number
  reasoning: string
  agent_domains: string[]
  candidates: Array<{ domain: string; score: number; matched_signals: string[] }>
}

interface SpecialistReport {
  domain: string
  summary?: string
  key_metrics?: string[]
  recommendation?: string
  error?: string
}

interface AgentPlanStep {
  step: number
  agent: string
  task: string
}

interface AgentReview {
  overall_quality: 'high' | 'medium' | 'low'
  issues: string[]
  approved: boolean
  suggested_improvements: string[]
}

interface AgentSecurity {
  risk_level: 'low' | 'medium' | 'high'
  findings: string[]
  blocked: boolean
  safe_to_show: boolean
}

interface AgentTrace {
  agent: string
  action: string
  timestamp: number
  input?: unknown
  output?: unknown
  error?: string | null
}

interface AgentResult {
  classification: DomainResult
  summary: string
  key_metrics: string[]
  recommendation: string
  participating_agents: string[]
  specialist_reports: SpecialistReport[]
  // Phase 4 fields
  plan?: AgentPlanStep[]
  review?: AgentReview
  security?: AgentSecurity
  traces?: AgentTrace[]
  error?: string
}

interface HistoryItem {
  id: string
  filename: string
  row_count: number
  column_count: number
  created_at: string
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ForecastColumnsInfo {
  numeric_columns: string[]
  date_column: string | null
}

interface ForecastResult {
  target_column: string
  x_axis_label: string
  trend: string
  slope_per_period: number
  r2_score: number
  mean_absolute_error: number
  history: { period: number; actual: number }[]
  forecast: { period: number; predicted: number }[]
  note: string
}

interface QualityReport {
  row_count: number
  column_count: number
  missing_values: Record<string, { count: number; pct: number }>
  duplicate_rows: number
  outliers: Record<string, { count: number; lower_bound: number; upper_bound: number }>
  type_issues: { column: string; detected_as: string; currently: string }[]
  quality_score: number
}

interface CleanReport {
  steps_applied: Array<Record<string, unknown>>
  before: { row_count: number; missing_values: number; duplicate_rows: number }
  after: { row_count: number; missing_values: number; duplicate_rows: number }
}

interface CleanResult {
  dataset_id: string
  filename: string
  report: CleanReport
  preview_rows: Record<string, string>[]
  columns: string[]
}

interface CorrelationResult {
  columns: string[]
  matrix: number[][]
}

interface ChartableColumns {
  numeric_columns: string[]
  categorical_columns: string[]
}

interface DistributionResult {
  type: 'histogram' | 'category'
  column: string
  bars: { label: string; value: number }[]
}

interface VizChart {
  chart_type: string
  x: string | null
  y: string | null
  title: string
  image_base64: string
}

interface VizGenerateResult {
  chart_type: string
  x: string | null
  y: string | null
  title: string | null
  image_base64: string
  interpreted: boolean
  reasoning: string | null
}

export type WorkspaceTab = 'upload' | 'profile' | 'analysis' | 'chat' | 'forecast' | 'clean' | 'eda' | 'visualize' | 'history'

interface UploadDatasetProps {
  activeTab: WorkspaceTab
  onTabChange: (tab: WorkspaceTab) => void
  onDatasetAvailabilityChange: (hasDataset: boolean) => void
}

async function authHeader() {
  const {
    data: { session },
  } = await supabase.auth.getSession()
  return { Authorization: `Bearer ${session?.access_token}` }
}

export default function UploadDataset({
  activeTab,
  onTabChange,
  onDatasetAvailabilityChange,
}: UploadDatasetProps) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const [detecting, setDetecting] = useState(false)
  const [domainError, setDomainError] = useState<string | null>(null)
  const [domainResult, setDomainResult] = useState<DomainResult | null>(null)

  const [analyzing, setAnalyzing] = useState(false)
  const [agentError, setAgentError] = useState<string | null>(null)
  const [agentResult, setAgentResult] = useState<AgentResult | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  const [forecastColumns, setForecastColumns] = useState<ForecastColumnsInfo | null>(null)
  const [forecastColumnsError, setForecastColumnsError] = useState<string | null>(null)
  const [selectedTarget, setSelectedTarget] = useState<string>('')
  const [forecasting, setForecasting] = useState(false)
  const [forecastError, setForecastError] = useState<string | null>(null)
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null)

  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [openingId, setOpeningId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null)
  const [qualityLoading, setQualityLoading] = useState(false)
  const [qualityError, setQualityError] = useState<string | null>(null)
  const [fillMissing, setFillMissing] = useState(true)
  const [missingStrategy, setMissingStrategy] = useState('mean')
  const [removeDuplicates, setRemoveDuplicates] = useState(true)
  const [fixTypes, setFixTypes] = useState(true)
  const [cleaning, setCleaning] = useState(false)
  const [cleanError, setCleanError] = useState<string | null>(null)
  const [cleanResult, setCleanResult] = useState<CleanResult | null>(null)

  const [correlation, setCorrelation] = useState<CorrelationResult | null>(null)
  const [correlationError, setCorrelationError] = useState<string | null>(null)
  const [correlationLoading, setCorrelationLoading] = useState(false)
  const [chartableColumns, setChartableColumns] = useState<ChartableColumns | null>(null)
  const [selectedChartColumn, setSelectedChartColumn] = useState<string>('')
  const [distribution, setDistribution] = useState<DistributionResult | null>(null)
  const [distributionLoading, setDistributionLoading] = useState(false)
  const [distributionError, setDistributionError] = useState<string | null>(null)

  const [vizChartType, setVizChartType] = useState('bar')
  const [vizX, setVizX] = useState('')
  const [vizY, setVizY] = useState('')
  const [vizNlRequest, setVizNlRequest] = useState('')
  const [vizLoading, setVizLoading] = useState(false)
  const [vizError, setVizError] = useState<string | null>(null)
  const [vizResult, setVizResult] = useState<VizGenerateResult | null>(null)
  const [dashboardCharts, setDashboardCharts] = useState<VizChart[] | null>(null)
  const [dashboardLoading, setDashboardLoading] = useState(false)
  const [dashboardError, setDashboardError] = useState<string | null>(null)

  function resetDownstreamState() {
    setDomainResult(null)
    setDomainError(null)
    setAgentResult(null)
    setAgentError(null)
    setMessages([])
    setChatError(null)
    setForecastColumns(null)
    setForecastColumnsError(null)
    setForecastResult(null)
    setForecastError(null)
    setSelectedTarget('')
    setQualityReport(null)
    setQualityError(null)
    setCleanResult(null)
    setCleanError(null)
    setCorrelation(null)
    setCorrelationError(null)
    setChartableColumns(null)
    setSelectedChartColumn('')
    setDistribution(null)
    setDistributionError(null)
    setVizResult(null)
    setVizError(null)
    setVizX('')
    setVizY('')
    setVizNlRequest('')
    setDashboardCharts(null)
    setDashboardError(null)
  }

  async function loadHistory() {
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/datasets`, { headers })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load history (${response.status})`)
      }
      setHistory(await response.json())
    } catch (err) {
      setHistoryError((err as Error).message)
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'history') {
      loadHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab])

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError(null)
    setResult(null)
    onDatasetAvailabilityChange(false)
    resetDownstreamState()

    try {
      const headers = await authHeader()
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/datasets/upload`, {
        method: 'POST',
        headers,
        body: formData,
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Upload failed (${response.status})`)
      }

      const data = await response.json()
      setResult(data)
      onDatasetAvailabilityChange(true)
      onTabChange('profile')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  async function handleDetectDomain() {
    if (!result) return
    setDetecting(true)
    setDomainError(null)
    setDomainResult(null)
    setAgentResult(null)
    setAgentError(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/domain/detect`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Detection failed (${response.status})`)
      }

      setDomainResult(await response.json())
    } catch (err) {
      setDomainError((err as Error).message)
    } finally {
      setDetecting(false)
    }
  }

  async function handleRunAnalysis() {
    if (!result || !domainResult) return
    setAnalyzing(true)
    setAgentError(null)
    setAgentResult(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/domain/analyze`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Analysis failed (${response.status})`)
      }

      setAgentResult(await response.json())
    } catch (err) {
      setAgentError((err as Error).message)
    } finally {
      setAnalyzing(false)
    }
  }

  async function handleAskQuestion(e: React.FormEvent) {
    e.preventDefault()
    if (!result || !question.trim() || asking) return

    const userMessage: ChatMessage = { role: 'user', content: question.trim() }
    const nextMessages = [...messages, userMessage]
    setMessages(nextMessages)
    setQuestion('')
    setAsking(true)
    setChatError(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: result.dataset_id,
          question: userMessage.content,
          history: nextMessages.slice(0, -1),
        }),
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Chat failed (${response.status})`)
      }

      const data = await response.json()
      setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }])
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    } catch (err) {
      setChatError((err as Error).message)
    } finally {
      setAsking(false)
    }
  }

  async function loadForecastColumns() {
    if (!result) return
    setForecastColumnsError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/forecast/columns`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load columns (${response.status})`)
      }
      const data = await response.json()
      setForecastColumns(data)
      if (data.numeric_columns?.length > 0) {
        setSelectedTarget(data.numeric_columns[0])
      }
    } catch (err) {
      setForecastColumnsError((err as Error).message)
    }
  }

  useEffect(() => {
    if (activeTab === 'forecast' && result && !forecastColumns) {
      loadForecastColumns()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, result])

  async function handleRunForecast() {
    if (!result || !selectedTarget) return
    setForecasting(true)
    setForecastError(null)
    setForecastResult(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/forecast`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id, target_column: selectedTarget, periods: 5 }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Forecast failed (${response.status})`)
      }
      setForecastResult(await response.json())
    } catch (err) {
      setForecastError((err as Error).message)
    } finally {
      setForecasting(false)
    }
  }

  async function loadQualityReport() {
    if (!result) return
    setQualityLoading(true)
    setQualityError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/clean/quality`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not check data quality (${response.status})`)
      }
      setQualityReport(await response.json())
    } catch (err) {
      setQualityError((err as Error).message)
    } finally {
      setQualityLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'clean' && result && !qualityReport && !qualityLoading) {
      loadQualityReport()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, result])

  async function handleApplyCleaning() {
    if (!result) return
    setCleaning(true)
    setCleanError(null)
    setCleanResult(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/clean/apply`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: result.dataset_id,
          fill_missing: fillMissing,
          missing_strategy: missingStrategy,
          remove_duplicates: removeDuplicates,
          fix_types: fixTypes,
        }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Cleaning failed (${response.status})`)
      }
      setCleanResult(await response.json())
    } catch (err) {
      setCleanError((err as Error).message)
    } finally {
      setCleaning(false)
    }
  }

  async function loadCorrelation() {
    if (!result) return
    setCorrelationLoading(true)
    setCorrelationError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/eda/correlation`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load correlation (${response.status})`)
      }
      setCorrelation(await response.json())
    } catch (err) {
      setCorrelationError((err as Error).message)
    } finally {
      setCorrelationLoading(false)
    }
  }

  async function loadChartableColumns() {
    if (!result) return
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/eda/columns`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })
      if (!response.ok) return
      const data: ChartableColumns = await response.json()
      setChartableColumns(data)
      const firstCol = data.numeric_columns[0] || data.categorical_columns[0] || ''
      setSelectedChartColumn(firstCol)
    } catch {
      // non-critical - chart column picker just stays empty
    }
  }

  useEffect(() => {
    if (activeTab === 'eda' && result && !correlation && !correlationLoading) {
      loadCorrelation()
      loadChartableColumns()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, result])

  async function loadDistribution() {
    if (!result || !selectedChartColumn) return
    setDistributionLoading(true)
    setDistributionError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/eda/distribution`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id, column: selectedChartColumn }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load chart (${response.status})`)
      }
      setDistribution(await response.json())
    } catch (err) {
      setDistributionError((err as Error).message)
    } finally {
      setDistributionLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'eda' && selectedChartColumn) {
      loadDistribution()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedChartColumn])

  async function handleOpenDataset(id: string) {
    setOpeningId(id)
    setError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/datasets/${id}`, { headers })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not open dataset (${response.status})`)
      }
      const data = await response.json()
      setResult(data)
      onDatasetAvailabilityChange(true)
      resetDownstreamState()
      onTabChange('profile')
    } catch (err) {
      setHistoryError((err as Error).message)
    } finally {
      setOpeningId(null)
    }
  }

  async function handleDeleteDataset(id: string) {
    setDeletingId(id)
    setHistoryError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/datasets/${id}`, { method: 'DELETE', headers })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not delete dataset (${response.status})`)
      }
      setHistory((prev) => prev.filter((h) => h.id !== id))
      if (result?.dataset_id === id) {
        setResult(null)
        onDatasetAvailabilityChange(false)
        resetDownstreamState()
      }
    } catch (err) {
      setHistoryError((err as Error).message)
    } finally {
      setDeletingId(null)
    }
  }

  async function handleGenerateChart(useNaturalLanguage: boolean) {
    if (!result) return
    setVizLoading(true)
    setVizError(null)
    setVizResult(null)
    try {
      const headers = await authHeader()
      const body: Record<string, unknown> = { dataset_id: result.dataset_id }
      if (useNaturalLanguage) {
        if (!vizNlRequest.trim()) {
          setVizError('Describe the chart you want first.')
          setVizLoading(false)
          return
        }
        body.nl_request = vizNlRequest.trim()
      } else {
        body.chart_type = vizChartType
        if (vizX) body.x = vizX
        if (vizY) body.y = vizY
      }

      const response = await fetch(`${API_BASE_URL}/viz/generate`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Chart generation failed (${response.status})`)
      }
      setVizResult(await response.json())
    } catch (err) {
      setVizError((err as Error).message)
    } finally {
      setVizLoading(false)
    }
  }

  async function handleGenerateDashboard() {
    if (!result) return
    setDashboardLoading(true)
    setDashboardError(null)
    setDashboardCharts(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/viz/dashboard`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Dashboard generation failed (${response.status})`)
      }
      const data = await response.json()
      setDashboardCharts(data.charts)
    } catch (err) {
      setDashboardError((err as Error).message)
    } finally {
      setDashboardLoading(false)
    }
  }

  const tabs: { id: WorkspaceTab; label: string; disabled?: boolean }[] = [
    { id: 'upload', label: 'Upload CSV' },
    { id: 'profile', label: 'Data profile', disabled: !result },
    { id: 'analysis', label: 'AI analysis', disabled: !result },
    { id: 'chat', label: 'Ask your data', disabled: !result },
    { id: 'forecast', label: 'Forecast', disabled: !result },
    { id: 'clean', label: 'Clean Data', disabled: !result },
    { id: 'eda', label: 'EDA & Charts', disabled: !result },
    { id: 'visualize', label: 'Visualize', disabled: !result },
    { id: 'history', label: 'History' },
  ]

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4 shadow-2xl shadow-black/10 sm:p-6">
      <div className="flex flex-col gap-4 border-b border-slate-800 pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Data studio</h2>
          <p className="mt-1 text-sm text-slate-400">
            Upload, profile, and analyze your data in dedicated workspace tabs.
          </p>
        </div>
        {result && <p className="text-sm font-medium text-cyan-300">{result.filename}</p>}
      </div>

      <div className="mt-5 flex gap-2 overflow-x-auto border-b border-slate-800 pb-3" role="tablist" aria-label="Dataset workspace">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            disabled={tab.disabled}
            onClick={() => onTabChange(tab.id)}
            className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-medium transition ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/30'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ---------- Upload tab ---------- */}
      {activeTab === 'upload' && (
        <div className="space-y-4 py-6">
          <div>
            <h3 className="text-lg font-medium text-white">Upload a dataset</h3>
            <p className="mt-1 text-sm text-slate-400">Choose a CSV or Excel (.xlsx) file to generate a data profile.</p>
          </div>
          <label className="block rounded-xl border border-dashed border-slate-700 bg-slate-950/30 p-6 transition hover:border-blue-500/70">
            <span className="sr-only">Choose CSV file</span>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileChange}
              disabled={uploading}
              className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-blue-600 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-blue-500 disabled:opacity-50"
            />
          </label>
          {uploading && <p className="text-sm text-slate-400">Uploading and profiling your dataset...</p>}
          {error && <ErrorBanner message={error} />}
        </div>
      )}

      {/* ---------- Profile tab ---------- */}
      {result && activeTab === 'profile' && (
        <div className="space-y-6 py-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Rows" value={result.row_count} />
            <Stat label="Columns" value={result.column_count} />
          </div>

          <div className="rounded-xl border border-slate-800 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="text-left p-2">Column</th>
                  <th className="text-left p-2">Type</th>
                  <th className="text-left p-2">Missing</th>
                  <th className="text-left p-2">Unique</th>
                  <th className="text-left p-2">Mean / Min / Max</th>
                </tr>
              </thead>
              <tbody>
                {result.columns.map((col) => (
                  <tr key={col.name} className="border-t border-slate-800">
                    <td className="p-2 font-medium">{col.name}</td>
                    <td className="p-2 text-slate-400">{col.dtype}</td>
                    <td className="p-2 text-slate-400">{col.missing_pct}%</td>
                    <td className="p-2 text-slate-400">{col.unique_count}</td>
                    <td className="p-2 text-slate-400">
                      {col.stats ? `${col.stats.mean} / ${col.stats.min} / ${col.stats.max}` : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <h3 className="text-sm font-medium mb-2 text-slate-400">Preview (first 10 rows)</h3>
            <div className="rounded-xl border border-slate-800 overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-slate-900 text-slate-400">
                  <tr>
                    {result.columns.map((col) => (
                      <th key={col.name} className="text-left p-2 whitespace-nowrap">
                        {col.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview_rows.map((row, i) => (
                    <tr key={i} className="border-t border-slate-800">
                      {result.columns.map((col) => (
                        <td key={col.name} className="p-2 whitespace-nowrap">
                          {row[col.name]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ---------- AI analysis tab ---------- */}
      {result && activeTab === 'analysis' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <h3 className="font-medium text-white">Collaborative AI analysis</h3>
            <p className="mt-1 text-sm text-slate-400">
              Route this dataset to the relevant specialists, then combine their evidence-grounded findings.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <button
              onClick={handleDetectDomain}
              disabled={detecting}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
            >
              {detecting ? 'Detecting...' : 'Detect Domain (AI)'}
            </button>

            {domainError && <ErrorBanner message={domainError} />}

            {domainResult && (
              <div className="rounded-xl border border-emerald-800 bg-emerald-950/30 p-4 space-y-3">
                <div>
                  <p className="text-sm text-slate-400">Primary domain</p>
                  <p className="text-2xl font-semibold text-emerald-400">{domainResult.primary_domain}</p>
                  <p className="text-xs text-slate-400 mt-1">
                    Routing confidence: {Math.round(domainResult.confidence * 100)}%
                  </p>
                  <p className="text-sm text-slate-300 mt-2">{domainResult.reasoning}</p>

                  {domainResult.secondary_domains.length > 0 && (
                    <p className="mt-3 text-sm text-slate-300">
                      <span className="text-slate-400">Also relevant: </span>
                      {domainResult.secondary_domains.join(', ')}
                    </p>
                  )}

                  {domainResult.tags.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {domainResult.tags.map((tag) => (
                        <span key={tag} className="rounded-full bg-emerald-900/50 px-2.5 py-1 text-xs text-emerald-200">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  onClick={handleRunAnalysis}
                  disabled={analyzing}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium hover:bg-indigo-500 disabled:opacity-50"
                >
                  {analyzing
                    ? 'Specialists are analyzing...'
                    : `Run collaborative analysis (${domainResult.agent_domains.join(' + ')})`}
                </button>
              </div>
            )}

            {agentError && <ErrorBanner message={agentError} />}
            {agentResult?.error && <ErrorBanner message={agentResult.error} />}

            {agentResult && (
              <div className="space-y-4">
                <div className="rounded-xl border border-indigo-800 bg-indigo-950/30 p-4 space-y-3">
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Participating specialists</p>
                    <div className="flex flex-wrap gap-2">
                      {(agentResult.participating_agents ?? []).map((agent) => (
                        <span key={agent} className="rounded-full bg-indigo-900/60 px-2.5 py-1 text-xs text-indigo-100">
                          {agent}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Primary perspective</p>
                    <p className="text-sm text-slate-200">{agentResult.summary}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Key Metrics</p>
                    <ul className="list-disc list-inside space-y-1">
                      {(agentResult.key_metrics ?? []).map((metric, i) => (
                        <li key={i} className="text-sm text-slate-200">
                          {metric}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Recommendation</p>
                    <p className="text-sm text-indigo-300">{agentResult.recommendation}</p>
                  </div>

                  {/* ---------- Phase 4: Plan / Review / Security ---------- */}
                  {(agentResult.plan || agentResult.review || agentResult.security) && (
                    <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4 space-y-3">
                      {agentResult.plan && agentResult.plan.length > 0 && (
                        <div>
                          <p className="text-sm text-slate-400 mb-1">Manager plan</p>
                          <div className="flex flex-wrap gap-2">
                            {agentResult.plan.map((step) => (
                              <span
                                key={step.step}
                                className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-200"
                              >
                                {step.step}. {step.agent} ({step.task})
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {agentResult.review && (
                        <div>
                          <p className="text-sm text-slate-400 mb-1">Reviewer</p>
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                                agentResult.review.overall_quality === 'high'
                                  ? 'bg-emerald-900/60 text-emerald-200'
                                  : agentResult.review.overall_quality === 'medium'
                                    ? 'bg-amber-900/60 text-amber-200'
                                    : 'bg-red-900/60 text-red-200'
                              }`}
                            >
                              Quality: {agentResult.review.overall_quality}
                            </span>
                            <span className="text-xs text-slate-400">
                              {agentResult.review.approved ? 'Approved' : 'Needs attention'}
                            </span>
                          </div>
                          {agentResult.review.issues?.length > 0 && (
                            <ul className="mt-2 list-disc list-inside text-xs text-amber-300">
                              {agentResult.review.issues.map((issue, i) => (
                                <li key={i}>{issue}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}

                      {agentResult.security && (
                        <div>
                          <p className="text-sm text-slate-400 mb-1">Security</p>
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                                agentResult.security.risk_level === 'low'
                                  ? 'bg-emerald-900/60 text-emerald-200'
                                  : agentResult.security.risk_level === 'medium'
                                    ? 'bg-amber-900/60 text-amber-200'
                                    : 'bg-red-900/60 text-red-200'
                              }`}
                            >
                              Risk: {agentResult.security.risk_level}
                            </span>
                            <span className="text-xs text-slate-400">
                              {agentResult.security.safe_to_show ? 'Safe to show' : 'Blocked'}
                            </span>
                          </div>
                          {agentResult.security.findings?.length > 0 && (
                            <ul className="mt-2 list-disc list-inside text-xs text-amber-300">
                              {agentResult.security.findings.map((f, i) => (
                                <li key={i}>{f}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {(agentResult.specialist_reports ?? []).length > 1 && (
                  <div className="grid gap-4 lg:grid-cols-2">
                    {(agentResult.specialist_reports ?? []).map((report) => (
                      <article key={report.domain} className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                        <p className="text-sm font-semibold text-cyan-300">{report.domain} specialist</p>
                        {report.error ? (
                          <p className="mt-2 text-sm text-amber-300">{report.error}</p>
                        ) : (
                          <>
                            <p className="mt-2 text-sm text-slate-300">{report.summary}</p>
                            {report.key_metrics && report.key_metrics.length > 0 && (
                              <ul className="mt-3 list-disc list-inside space-y-1 text-sm text-slate-300">
                                {report.key_metrics.map((metric, index) => (
                                  <li key={index}>{metric}</li>
                                ))}
                              </ul>
                            )}
                            <p className="mt-3 text-sm text-indigo-300">{report.recommendation}</p>
                          </>
                        )}
                      </article>
                    ))}
                  </div>
                )}

                {/* ---------- Phase 4: Agent Traces ---------- */}
                {agentResult.traces && agentResult.traces.length > 0 && (
                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                    <p className="text-sm font-medium text-slate-300 mb-3">Agent traces</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto text-xs">
                      {agentResult.traces.map((t, i) => (
                        <div
                          key={i}
                          className="flex flex-wrap gap-2 rounded-lg bg-slate-900/80 px-3 py-2"
                        >
                          <span className="font-medium text-cyan-300">{t.agent}</span>
                          <span className="text-slate-400">→ {t.action}</span>
                          {t.error && <span className="text-red-400">Error: {t.error}</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---------- Chat tab ---------- */}
      {result && activeTab === 'chat' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <h3 className="font-medium text-white">Ask your data</h3>
            <p className="mt-1 text-sm text-slate-400">
              Ask questions in plain English. Answers use the exact, full-dataset statistics - never guesses from a sample.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 flex flex-col h-[420px]">
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.length === 0 && (
                <p className="text-sm text-slate-500">
                  Try asking something like "What's the average value in [a numeric column]?" or "Which category
                  appears most often?"
                </p>
              )}
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${
                    msg.role === 'user' ? 'ml-auto bg-blue-600 text-white' : 'bg-slate-800 text-slate-200'
                  }`}
                >
                  {msg.content}
                </div>
              ))}
              {asking && (
                <div className="max-w-[85%] rounded-xl px-3 py-2 text-sm bg-slate-800 text-slate-400">Thinking...</div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form onSubmit={handleAskQuestion} className="border-t border-slate-800 p-3 flex gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about this dataset..."
                disabled={asking}
                className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-blue-500 disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={asking || !question.trim()}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
              >
                Ask
              </button>
            </form>
          </div>

          {chatError && <ErrorBanner message={chatError} />}
        </div>
      )}

      {/* ---------- Forecast tab ---------- */}
      {result && activeTab === 'forecast' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <h3 className="font-medium text-white">Forecast a numeric column</h3>
            <p className="mt-1 text-sm text-slate-400">
              Trains a real scikit-learn linear regression model on the full dataset - not an AI guess.
            </p>
          </div>

          {forecastColumnsError && <ErrorBanner message={forecastColumnsError} />}

          {forecastColumns && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <label className="text-sm text-slate-400">Column to forecast:</label>
                <select
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-blue-500"
                >
                  {forecastColumns.numeric_columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleRunForecast}
                  disabled={forecasting || !selectedTarget}
                  className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {forecasting ? 'Training model...' : 'Run Forecast'}
                </button>
              </div>

              {forecastColumns.date_column && (
                <p className="text-xs text-slate-500">
                  Using "{forecastColumns.date_column}" as the time axis.
                </p>
              )}
            </div>
          )}

          {forecastError && <ErrorBanner message={forecastError} />}

          {forecastResult && (
            <div className="rounded-xl border border-blue-800 bg-blue-950/30 p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Stat label="Trend" value={forecastResult.trend} isText />
                <Stat label="R\u00b2 fit" value={forecastResult.r2_score} />
                <Stat label="Avg error" value={forecastResult.mean_absolute_error} />
              </div>

              <p className="text-xs text-slate-400">{forecastResult.note}</p>

              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-2">
                  Predicted next {forecastResult.forecast.length} periods ({forecastResult.x_axis_label})
                </h4>
                <div className="rounded-xl border border-slate-800 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-900 text-slate-400">
                      <tr>
                        <th className="text-left p-2">Period</th>
                        <th className="text-left p-2">Predicted {forecastResult.target_column}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forecastResult.forecast.map((row) => (
                        <tr key={row.period} className="border-t border-slate-800">
                          <td className="p-2 text-slate-400">{row.period}</td>
                          <td className="p-2 font-medium text-blue-300">{row.predicted}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---------- Clean Data tab ---------- */}
      {result && activeTab === 'clean' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <h3 className="font-medium text-white">Data quality &amp; cleaning</h3>
            <p className="mt-1 text-sm text-slate-400">
              The original upload is never overwritten - cleaning saves a separate cleaned copy.
            </p>
          </div>

          {qualityLoading && <p className="text-sm text-slate-400">Checking data quality...</p>}
          {qualityError && <ErrorBanner message={qualityError} />}

          {qualityReport && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-400">Data quality score</p>
                <p className="text-2xl font-semibold text-white">{qualityReport.quality_score}/100</p>
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Rows" value={qualityReport.row_count} />
                <Stat label="Duplicates" value={qualityReport.duplicate_rows} />
                <Stat label="Columns w/ missing" value={Object.keys(qualityReport.missing_values).length} />
                <Stat label="Columns w/ outliers" value={Object.keys(qualityReport.outliers).length} />
              </div>

              {qualityReport.type_issues.length > 0 && (
                <p className="text-xs text-amber-300">
                  {qualityReport.type_issues.length} column(s) look like they should be numeric or date but are
                  stored as text.
                </p>
              )}
            </div>
          )}

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <p className="text-sm font-medium text-slate-300">Choose cleaning steps</p>

            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={fillMissing} onChange={(e) => setFillMissing(e.target.checked)} />
              Fill missing values
            </label>
            {fillMissing && (
              <select
                value={missingStrategy}
                onChange={(e) => setMissingStrategy(e.target.value)}
                className="ml-6 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-blue-500"
              >
                <option value="mean">Numeric: mean, Text: most common value</option>
                <option value="median">Numeric: median, Text: most common value</option>
                <option value="drop_rows">Drop rows with any missing value</option>
              </select>
            )}

            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={removeDuplicates}
                onChange={(e) => setRemoveDuplicates(e.target.checked)}
              />
              Remove duplicate rows
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={fixTypes} onChange={(e) => setFixTypes(e.target.checked)} />
              Fix column types (text that's really numbers/dates)
            </label>

            <button
              onClick={handleApplyCleaning}
              disabled={cleaning}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
            >
              {cleaning ? 'Cleaning...' : 'Apply Cleaning'}
            </button>
          </div>

          {cleanError && <ErrorBanner message={cleanError} />}

          {cleanResult && (
            <div className="rounded-xl border border-emerald-800 bg-emerald-950/30 p-4 space-y-4">
              <h4 className="text-sm font-medium text-emerald-300">Before vs After</h4>
              <div className="rounded-xl border border-slate-800 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="text-left p-2"></th>
                      <th className="text-left p-2">Before</th>
                      <th className="text-left p-2">After</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-400">Rows</td>
                      <td className="p-2">{cleanResult.report.before.row_count}</td>
                      <td className="p-2 text-emerald-300">{cleanResult.report.after.row_count}</td>
                    </tr>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-400">Missing values</td>
                      <td className="p-2">{cleanResult.report.before.missing_values}</td>
                      <td className="p-2 text-emerald-300">{cleanResult.report.after.missing_values}</td>
                    </tr>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-400">Duplicate rows</td>
                      <td className="p-2">{cleanResult.report.before.duplicate_rows}</td>
                      <td className="p-2 text-emerald-300">{cleanResult.report.after.duplicate_rows}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-2">Cleaned data preview</h4>
                <div className="rounded-xl border border-slate-800 overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-900 text-slate-400">
                      <tr>
                        {cleanResult.columns.map((col) => (
                          <th key={col} className="text-left p-2 whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {cleanResult.preview_rows.map((row, i) => (
                        <tr key={i} className="border-t border-slate-800">
                          {cleanResult.columns.map((col) => (
                            <td key={col} className="p-2 whitespace-nowrap">
                              {row[col]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---------- EDA & Charts tab ---------- */}
      {result && activeTab === 'eda' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <h3 className="font-medium text-white">Correlation &amp; distributions</h3>
            <p className="mt-1 text-sm text-slate-400">
              Computed directly from the full dataset with pandas - no AI involved.
            </p>
          </div>

          {correlationLoading && <p className="text-sm text-slate-400">Computing correlation matrix...</p>}
          {correlationError && <ErrorBanner message={correlationError} />}

          {correlation && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 overflow-x-auto">
              <h4 className="text-sm font-medium text-slate-300 mb-3">Correlation matrix</h4>
              <table className="text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2"></th>
                    {correlation.columns.map((col) => (
                      <th key={col} className="p-2 text-slate-400 whitespace-nowrap font-medium">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlation.matrix.map((row, i) => (
                    <tr key={i}>
                      <td className="p-2 text-slate-400 whitespace-nowrap font-medium">{correlation.columns[i]}</td>
                      {row.map((value, j) => (
                        <td
                          key={j}
                          className="p-2 text-center whitespace-nowrap"
                          style={{
                            backgroundColor:
                              value >= 0
                                ? `rgba(59, 130, 246, ${Math.abs(value) * 0.6})`
                                : `rgba(239, 68, 68, ${Math.abs(value) * 0.6})`,
                          }}
                        >
                          {value.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-slate-500">
                Blue = positive correlation, red = negative. Darker = stronger relationship.
              </p>
            </div>
          )}

          {chartableColumns && (chartableColumns.numeric_columns.length > 0 || chartableColumns.categorical_columns.length > 0) && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <label className="text-sm text-slate-400">Chart a column:</label>
                <select
                  value={selectedChartColumn}
                  onChange={(e) => setSelectedChartColumn(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-blue-500"
                >
                  {chartableColumns.numeric_columns.length > 0 && (
                    <optgroup label="Numeric (histogram)">
                      {chartableColumns.numeric_columns.map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  {chartableColumns.categorical_columns.length > 0 && (
                    <optgroup label="Categorical (bar chart)">
                      {chartableColumns.categorical_columns.map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </div>

              {distributionLoading && <p className="text-sm text-slate-400">Loading chart...</p>}
              {distributionError && <ErrorBanner message={distributionError} />}

              {distribution && (
                <div className="space-y-2">
                  <p className="text-xs text-slate-500">
                    {distribution.type === 'histogram' ? 'Value ranges' : 'Top categories'} for{' '}
                    {distribution.column}
                  </p>
                  {(() => {
                    const maxValue = Math.max(...distribution.bars.map((b) => b.value), 1)
                    return distribution.bars.map((bar) => (
                      <div key={bar.label} className="flex items-center gap-2 text-xs">
                        <span className="w-32 shrink-0 truncate text-slate-400" title={bar.label}>
                          {bar.label}
                        </span>
                        <div className="flex-1 rounded bg-slate-800 h-5 overflow-hidden">
                          <div
                            className="h-full rounded bg-blue-600"
                            style={{ width: `${(bar.value / maxValue) * 100}%` }}
                          />
                        </div>
                        <span className="w-10 shrink-0 text-right text-slate-300">{bar.value}</span>
                      </div>
                    ))
                  })()}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ---------- Visualize tab ---------- */}
      {result && activeTab === 'visualize' && (
        <div className="space-y-6 py-6">
          <div>
            <h3 className="text-lg font-medium text-white">Visualization Agent</h3>
            <p className="mt-1 text-sm text-slate-400">
              Build a chart manually, describe one in plain language, or auto-generate a dashboard.
            </p>
          </div>

          {/* Manual chart builder */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-4">
            <h4 className="text-sm font-medium text-slate-200">Build a chart</h4>
            <div className="grid gap-4 sm:grid-cols-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Chart type</label>
                <select
                  value={vizChartType}
                  onChange={(e) => setVizChartType(e.target.value)}
                  className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
                >
                  <option value="histogram">Histogram</option>
                  <option value="bar">Bar</option>
                  <option value="line">Line</option>
                  <option value="scatter">Scatter</option>
                  <option value="heatmap">Heatmap (correlation)</option>
                </select>
              </div>
              {vizChartType !== 'heatmap' && (
                <div>
                  <label className="block text-xs text-slate-400 mb-1">X column</label>
                  <select
                    value={vizX}
                    onChange={(e) => setVizX(e.target.value)}
                    className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
                  >
                    <option value="">Select column</option>
                    {result.columns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {(vizChartType === 'bar' || vizChartType === 'line' || vizChartType === 'scatter') && (
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Y column</label>
                  <select
                    value={vizY}
                    onChange={(e) => setVizY(e.target.value)}
                    className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
                  >
                    <option value="">Select column</option>
                    {result.columns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <div className="flex items-end">
                <button
                  onClick={() => handleGenerateChart(false)}
                  disabled={vizLoading}
                  className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
                >
                  {vizLoading ? 'Generating...' : 'Generate chart'}
                </button>
              </div>
            </div>
          </div>

          {/* Natural language chart request */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <h4 className="text-sm font-medium text-slate-200">Or describe the chart you want</h4>
            <div className="flex gap-2">
              <input
                type="text"
                value={vizNlRequest}
                onChange={(e) => setVizNlRequest(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !vizLoading) handleGenerateChart(true)
                }}
                placeholder='e.g. "show revenue by region as a bar chart"'
                className="flex-1 rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500"
              />
              <button
                onClick={() => handleGenerateChart(true)}
                disabled={vizLoading}
                className="rounded-lg bg-purple-600 px-4 py-2 text-sm font-medium hover:bg-purple-500 disabled:opacity-50 shrink-0"
              >
                {vizLoading ? 'Thinking...' : 'Generate'}
              </button>
            </div>
          </div>

          {vizError && <ErrorBanner message={vizError} />}

          {vizResult && (
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4 space-y-2">
              {vizResult.interpreted && vizResult.reasoning && (
                <p className="text-xs text-purple-300">AI interpretation: {vizResult.reasoning}</p>
              )}
              <img
                src={`data:image/png;base64,${vizResult.image_base64}`}
                alt={vizResult.title || 'Generated chart'}
                className="w-full rounded-lg border border-slate-800"
              />
            </div>
          )}

          {/* Auto dashboard */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-slate-200">Auto-generated dashboard</h4>
              <button
                onClick={handleGenerateDashboard}
                disabled={dashboardLoading}
                className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
              >
                {dashboardLoading ? 'Building dashboard...' : 'Generate dashboard'}
              </button>
            </div>

            {dashboardError && <ErrorBanner message={dashboardError} />}

            {dashboardCharts && (
              <div className="grid gap-4 sm:grid-cols-2">
                {dashboardCharts.map((chart, i) => (
                  <div key={i} className="rounded-lg border border-slate-800 overflow-hidden">
                    <img
                      src={`data:image/png;base64,${chart.image_base64}`}
                      alt={chart.title}
                      className="w-full"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---------- History tab ---------- */}
      {activeTab === 'history' && (
        <div className="space-y-4 py-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-white">Your datasets</h3>
            <button
              onClick={loadHistory}
              disabled={historyLoading}
              className="text-xs text-slate-400 hover:text-slate-200 disabled:opacity-50"
            >
              {historyLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>

          {historyError && <ErrorBanner message={historyError} />}

          {historyLoading && history.length === 0 && (
            <p className="text-sm text-slate-400">Loading your datasets...</p>
          )}

          {!historyLoading && history.length === 0 && !historyError && (
            <div className="rounded-xl border border-dashed border-slate-700 p-8 text-center">
              <p className="text-sm text-slate-400">No datasets yet. Upload a CSV to see it here.</p>
            </div>
          )}

          {history.length > 0 && (
            <div className="rounded-xl border border-slate-800 overflow-hidden divide-y divide-slate-800">
              {history.map((item) => (
                <div key={item.id} className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-100">{item.filename}</p>
                    <p className="text-xs text-slate-500">
                      {item.row_count} rows &middot; {item.column_count} columns &middot;{' '}
                      {new Date(item.created_at).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleOpenDataset(item.id)}
                      disabled={openingId === item.id}
                      className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium hover:bg-blue-500 disabled:opacity-50"
                    >
                      {openingId === item.id ? 'Opening...' : 'Open'}
                    </button>
                    <button
                      onClick={() => handleDeleteDataset(item.id)}
                      disabled={deletingId === item.id}
                      className="rounded-lg bg-red-600/80 px-3 py-1.5 text-xs font-medium hover:bg-red-600 disabled:opacity-50"
                    >
                      {deletingId === item.id ? 'Deleting...' : 'Delete'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-800 bg-red-950/30 px-3 py-2 text-sm text-red-300">{message}</div>
  )
}

function Stat({ label, value, isText }: { label: string; value: number | string; isText?: boolean }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <p className="text-xs text-slate-400">{label}</p>
      <p className={`font-semibold ${isText ? 'text-lg capitalize' : 'text-2xl'}`}>{value}</p>
    </div>
  )
}