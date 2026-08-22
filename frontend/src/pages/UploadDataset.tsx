import { useEffect, useRef, useState } from 'react'
import _createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import { supabase } from '../lib/supabaseClient'

// FIX: Unwrap the function if Vite nested it inside a '.default' property
const createPlotlyComponent = (_createPlotlyComponent as any).default || _createPlotlyComponent
const Plot = createPlotlyComponent(Plotly as any)

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

interface ReportSection {
  heading: string
  body?: string
  bullets?: string[]
  recommendation?: string
}

interface ReportTimelineStep {
  agent: string
  role: string
  detail: string
}

interface ReportChart {
  title: string
  image_base64: string
}

interface ReportContent {
  title: string
  subtitle?: string
  dataset_id: string
  filename: string
  primary_domain: string
  generated_at: string
  executive_summary: string
  agent_collaboration_narrative?: string
  agent_timeline?: ReportTimelineStep[]
  key_metrics: string[]
  recommendation: string
  sections: ReportSection[]
  charts?: ReportChart[]
  participating_agents: string[]
  review?: AgentReview
  security?: AgentSecurity
  approval_status: 'pending' | 'approved' | 'rejected'
  id: string
  version_number: number
  rejection_reason?: string | null
}

interface ReportVersion extends ReportContent {
  created_at?: string
}


interface AgentStreamEvent {
  type: string
  agent: string
  message: string
  data?: Record<string, unknown> | null
  timestamp: number
}

interface AgentStreamDone {
  type: '__done__'
  workflow_id: string
  status: string
  result: {
    summary?: string
    recommendation?: string
    key_metrics?: string[]
    participating_agents?: string[]
    goal?: string
  }
  approval?: AgentApproval | null
}

interface AgentApproval {
  id: string
  resource_type: string
  resource_id: string
  dataset_id?: string | null
  version_number: number
  approval_status: 'pending' | 'approved' | 'rejected'
  rejection_reason?: string | null
  created_at?: string
}

interface AgentRunSummary {
  workflow_id: string
  dataset_id: string | null
  user_query: string
  goal: string
  status: string
  created_at: string
  updated_at: string
}

interface AgentRunDetail {
  workflow_id: string
  user_id: string
  dataset_id: string | null
  user_query: string
  goal: string
  status: string
  state: {
    final_output?: AgentStreamDone['result']
  }
  created_at: string
  updated_at: string
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

interface PlotlyFigure {
  data: unknown[]
  layout: Record<string, unknown>
}

interface VizChart {
  chart_type: string
  x: string | null
  y: string | null
  title: string
  figure: PlotlyFigure
  insight?: string
  reasoning?: string | null
}

interface VizGenerateResponse {
  charts: VizChart[]
  interpreted: boolean
}

export type WorkspaceTab = 'upload' | 'profile' | 'analysis' | 'agent' | 'chat' | 'forecast' | 'clean' | 'eda' | 'visualize' | 'history'

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

// Turns a free-text metric string like "Mean daily unlocks of 171.45"
// into a compact { value, unit, label } shape for stat cards.
function parseMetricCard(metric: string): { value: string; unit: string; label: string } {
  const numMatch = metric.match(/[\d][\d,]*\.?\d*/)
  const value = numMatch ? numMatch[0] : ''

  const lower = metric.toLowerCase()
  let unit = ''
  if (lower.includes('hour')) unit = 'hrs'
  else if (lower.includes('%') || lower.includes('percent')) unit = '%'
  else if (lower.includes('case')) unit = 'cases'
  else if (lower.includes('user')) unit = 'users'
  else if (lower.includes('day')) unit = '/day'

  const fillers = new Set([
    'mean', 'average', 'avg', 'total', 'of', 'reporting', 'individuals',
    'participants', 'students', 'levels', 'level', 'per', 'day', 'hours',
    'hour', 'cases', 'a', 'the', 'with', 'users', 'user', '%', 'percent',
  ])

  const label = metric
    .replace(numMatch ? numMatch[0] : '', '')
    .replace(/[,%]/g, ' ')
    .split(/\s+/)
    .filter((w) => w && !fillers.has(w.toLowerCase()))
    .slice(0, 3)
    .join(' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())

  return { value, unit, label: label || metric.slice(0, 28) }
}

const STAT_ICONS = [
  <svg key="0" viewBox="0 0 24 24" fill="none" className="h-4 w-4">
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" />
    <path d="M12 7v5l3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>,
  <svg key="1" viewBox="0 0 24 24" fill="none" className="h-4 w-4">
    <rect x="5" y="11" width="14" height="9" rx="2" stroke="currentColor" strokeWidth="1.5" />
    <path d="M8 11V8a4 4 0 0 1 8 0v3" stroke="currentColor" strokeWidth="1.5" />
  </svg>,
  <svg key="2" viewBox="0 0 24 24" fill="none" className="h-4 w-4">
    <path d="M12 3l9 16H3l9-16z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    <path d="M12 10v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="12" cy="17" r="0.5" fill="currentColor" />
  </svg>,
  <svg key="3" viewBox="0 0 24 24" fill="none" className="h-4 w-4">
    <path d="M4 19V5a1 1 0 0 1 1-1h6v16" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    <path d="M11 4h8a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-8" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>,
]

// Small icon set used by the "System routing & execution details" panel.
function domainIcon(domain: string) {
  const key = domain.toLowerCase()
  if (key.includes('health')) {
    return (
      <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
        <path
          d="M12 21s-7-4.35-9.5-8.5C.5 8.5 3 5 6.5 5c2 0 3.5 1.2 4.5 2.5C12 6.2 13.5 5 15.5 5 19 5 21.5 8.5 20 12.5 17.5 16.65 12 21 12 21z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    )
  }
  if (key.includes('educ')) {
    return (
      <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
        <path d="M12 3l9 5-9 5-9-5 9-5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <path
          d="M6 10.5V16c0 1 2.7 2.5 6 2.5s6-1.5 6-2.5v-5.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    )
  }
  if (key.includes('financ')) {
    return (
      <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" />
        <path
          d="M12 7v10M9 9.5c0-1 1-1.5 3-1.5s3 .8 3 2-1.3 1.7-3 2-3 .9-3 2 1.3 2 3 2 3-.5 3-1.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    )
  }
  if (key.includes('hr')) {
    return (
      <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
        <circle cx="9" cy="8" r="3" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="17" cy="9" r="2.5" stroke="currentColor" strokeWidth="1.5" />
        <path
          d="M3 20c0-3 2.7-5 6-5s6 2 6 5M15 20c0-2.2 1.6-4 4-4.3"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    )
  }
  if (key.includes('retail')) {
    return (
      <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
        <path
          d="M4 8h16l-1.5 10a2 2 0 0 1-2 1.7H7.5a2 2 0 0 1-2-1.7L4 8z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
        <path d="M8 8V6a4 4 0 0 1 8 0v2" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
      <path
        d="M4 19V9M10 19V5M16 19v-7M4 19h16"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function DatasetIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5">
      <ellipse cx="12" cy="6" rx="7" ry="2.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5 6v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5 12v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

function OrchestratorIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5">
      <rect x="7" y="7" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M12 2v3M12 19v3M2 12h3M19 12h3M4.5 4.5l2 2M17.5 17.5l2 2M19.5 4.5l-2 2M6.5 17.5l-2 2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4 shrink-0 text-slate-600">
      <path d="M4 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
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
  const [activeSpecialistTab, setActiveSpecialistTab] = useState<string | null>(null)
  const [showSystemDetails, setShowSystemDetails] = useState(false)

  const [reportGenerating, setReportGenerating] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)
  const [reportContent, setReportContent] = useState<ReportContent | null>(null)
  const [reportDownloading, setReportDownloading] = useState<'pdf' | 'docx' | null>(null)
  const [reportEditingSummary, setReportEditingSummary] = useState(false)
  const [showAgentDiscussion, setShowAgentDiscussion] = useState(false)

  const [agentQuery, setAgentQuery] = useState('')
  const [agentStreamRunning, setAgentStreamRunning] = useState(false)
  const [agentStreamEvents, setAgentStreamEvents] = useState<AgentStreamEvent[]>([])
  const [agentStreamResult, setAgentStreamResult] = useState<AgentStreamDone | null>(null)
  const [agentStreamError, setAgentStreamError] = useState<string | null>(null)
  const [agentApproval, setAgentApproval] = useState<AgentApproval | null>(null)
  const [agentApprovalActionLoading, setAgentApprovalActionLoading] = useState<'approve' | 'reject' | null>(null)
  const [agentApprovalError, setAgentApprovalError] = useState<string | null>(null)
  const [agentShowRejectForm, setAgentShowRejectForm] = useState(false)
  const [agentRejectReason, setAgentRejectReason] = useState('')

  const [agentHistoryOpen, setAgentHistoryOpen] = useState(false)
  const [agentHistoryLoading, setAgentHistoryLoading] = useState(false)
  const [agentHistoryError, setAgentHistoryError] = useState<string | null>(null)
  const [agentHistoryRuns, setAgentHistoryRuns] = useState<AgentRunSummary[]>([])
  const [selectedHistoryRun, setSelectedHistoryRun] = useState<AgentRunDetail | null>(null)
  const [selectedHistoryApproval, setSelectedHistoryApproval] = useState<AgentApproval | null>(null)
  const [historyLoadingId, setHistoryLoadingId] = useState<string | null>(null)
  const [historyApprovalActionLoading, setHistoryApprovalActionLoading] = useState<'approve' | 'reject' | null>(null)
  const [historyShowRejectForm, setHistoryShowRejectForm] = useState(false)
  const [historyRejectReason, setHistoryRejectReason] = useState('')
  const [reportVersions, setReportVersions] = useState<ReportVersion[]>([])
  const [showVersionHistory, setShowVersionHistory] = useState(false)
  const [versionsLoading, setVersionsLoading] = useState(false)
  const [showRejectForm, setShowRejectForm] = useState(false)
  const [rejectReason, setRejectReason] = useState('')
  const [reportActionLoading, setReportActionLoading] = useState<'approve' | 'reject' | 'resubmit' | null>(null)

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
  const [vizResult, setVizResult] = useState<VizGenerateResponse | null>(null)
  const [dashboardCharts, setDashboardCharts] = useState<VizChart[] | null>(null)
  const [dashboardLoading, setDashboardLoading] = useState(false)
  const [dashboardError, setDashboardError] = useState<string | null>(null)

  function resetDownstreamState() {
    setDomainResult(null)
    setDomainError(null)
    setAgentResult(null)
    setAgentError(null)
    setReportContent(null)
    setReportError(null)
    setReportEditingSummary(false)
    setShowAgentDiscussion(false)
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

  async function handleRunAgentStream() {
    if (!result || !agentQuery.trim()) return
    setAgentStreamRunning(true)
    setAgentStreamEvents([])
    setAgentStreamResult(null)
    setAgentStreamError(null)
    setAgentApproval(null)
    setAgentApprovalError(null)
    setAgentShowRejectForm(false)
    setAgentRejectReason('')

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/agent/run/stream`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: agentQuery.trim(), dataset_id: result.dataset_id }),
      })

      if (!response.ok || !response.body) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Failed to start (${response.status})`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''

        for (const chunk of chunks) {
          const line = chunk.trim()
          if (!line.startsWith('data:')) continue
          const jsonStr = line.slice(5).trim()
          if (!jsonStr) continue

          const event = JSON.parse(jsonStr)
          if (event.type === '__done__') {
            setAgentStreamResult(event)
            setAgentApproval(event.approval ?? null)
          } else if (event.type === '__error__') {
            setAgentStreamError(event.message)
          } else {
            setAgentStreamEvents((prev) => [...prev, event])
          }
        }
      }
    } catch (err) {
      setAgentStreamError((err as Error).message)
    } finally {
      setAgentStreamRunning(false)
    }
  }

  async function handleApproveAgentWorkflow() {
    if (!agentApproval) return
    setAgentApprovalActionLoading('approve')
    setAgentApprovalError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/approvals/${agentApproval.id}/approve`, {
        method: 'POST',
        headers,
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Approve failed (${response.status})`)
      }
      setAgentApproval(await response.json())
    } catch (err) {
      setAgentApprovalError((err as Error).message)
    } finally {
      setAgentApprovalActionLoading(null)
    }
  }

  async function handleRejectAgentWorkflow() {
    if (!agentApproval || !agentRejectReason.trim()) return
    setAgentApprovalActionLoading('reject')
    setAgentApprovalError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/approvals/${agentApproval.id}/reject`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: agentRejectReason.trim() }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Reject failed (${response.status})`)
      }
      setAgentApproval(await response.json())
      setAgentShowRejectForm(false)
      setAgentRejectReason('')
    } catch (err) {
      setAgentApprovalError((err as Error).message)
    } finally {
      setAgentApprovalActionLoading(null)
    }
  }

  async function handleToggleAgentHistory() {
    const opening = !agentHistoryOpen
    setAgentHistoryOpen(opening)
    if (opening && result) {
      setAgentHistoryLoading(true)
      setAgentHistoryError(null)
      try {
        const headers = await authHeader()
        const response = await fetch(`${API_BASE_URL}/agent/runs?dataset_id=${encodeURIComponent(result.dataset_id)}`, {
          headers,
        })
        if (!response.ok) {
          const errBody = await response.json().catch(() => ({}))
          throw new Error(errBody.detail || `Failed to load history (${response.status})`)
        }
        const data = await response.json()
        setAgentHistoryRuns(data.runs || [])
      } catch (err) {
        setAgentHistoryError((err as Error).message)
      } finally {
        setAgentHistoryLoading(false)
      }
    }
  }

  async function handleOpenHistoryRun(workflowId: string) {
    setHistoryLoadingId(workflowId)
    setAgentHistoryError(null)
    setSelectedHistoryRun(null)
    setSelectedHistoryApproval(null)
    setHistoryShowRejectForm(false)
    setHistoryRejectReason('')
    try {
      const headers = await authHeader()
      const [runResponse, approvalResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/agent/runs/${workflowId}`, { headers }),
        fetch(`${API_BASE_URL}/approvals/agent_workflow/${workflowId}`, { headers }),
      ])
      if (!runResponse.ok) {
        const errBody = await runResponse.json().catch(() => ({}))
        throw new Error(errBody.detail || `Failed to load run (${runResponse.status})`)
      }
      setSelectedHistoryRun(await runResponse.json())
      if (approvalResponse.ok) {
        const approvalData = await approvalResponse.json()
        setSelectedHistoryApproval(approvalData.versions?.[0] ?? null)
      }
    } catch (err) {
      setAgentHistoryError((err as Error).message)
    } finally {
      setHistoryLoadingId(null)
    }
  }

  async function handleApproveHistoryRun() {
    if (!selectedHistoryApproval) return
    setHistoryApprovalActionLoading('approve')
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/approvals/${selectedHistoryApproval.id}/approve`, {
        method: 'POST',
        headers,
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Approve failed (${response.status})`)
      }
      setSelectedHistoryApproval(await response.json())
    } catch (err) {
      setAgentHistoryError((err as Error).message)
    } finally {
      setHistoryApprovalActionLoading(null)
    }
  }

  async function handleRejectHistoryRun() {
    if (!selectedHistoryApproval || !historyRejectReason.trim()) return
    setHistoryApprovalActionLoading('reject')
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/approvals/${selectedHistoryApproval.id}/reject`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: historyRejectReason.trim() }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Reject failed (${response.status})`)
      }
      setSelectedHistoryApproval(await response.json())
      setHistoryShowRejectForm(false)
      setHistoryRejectReason('')
    } catch (err) {
      setAgentHistoryError((err as Error).message)
    } finally {
      setHistoryApprovalActionLoading(null)
    }
  }

  async function handleGenerateReport() {
    if (!result || !agentResult || agentResult.error) return
    setReportGenerating(true)
    setReportError(null)
    setReportContent(null)
    setShowRejectForm(false)
    setRejectReason('')

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/report/generate`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: result.dataset_id,
          filename: result.filename,
          analysis: agentResult,
        }),
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Report generation failed (${response.status})`)
      }

      const generated: ReportContent = await response.json()
      setReportContent(generated)
      setReportVersions((prev) => [generated, ...prev])
    } catch (err) {
      setReportError((err as Error).message)
    } finally {
      setReportGenerating(false)
    }
  }

  async function handleApproveReport() {
    if (!reportContent) return
    setReportActionLoading('approve')
    setReportError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/report/${reportContent.id}/approve`, {
        method: 'POST',
        headers,
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Approve failed (${response.status})`)
      }
      const updated: ReportContent = await response.json()
      setReportContent(updated)
      setReportVersions((prev) => prev.map((v) => (v.id === updated.id ? updated : v)))
    } catch (err) {
      setReportError((err as Error).message)
    } finally {
      setReportActionLoading(null)
    }
  }

  async function handleRejectReport() {
    if (!reportContent || !rejectReason.trim()) return
    setReportActionLoading('reject')
    setReportError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/report/${reportContent.id}/reject`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: rejectReason.trim() }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Reject failed (${response.status})`)
      }
      const updated: ReportContent = await response.json()
      setReportContent(updated)
      setReportVersions((prev) => prev.map((v) => (v.id === updated.id ? updated : v)))
      setShowRejectForm(false)
      setRejectReason('')
    } catch (err) {
      setReportError((err as Error).message)
    } finally {
      setReportActionLoading(null)
    }
  }

  async function handleResubmitReport() {
    if (!reportContent) return
    setReportActionLoading('resubmit')
    setReportError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/report/${reportContent.id}/resubmit`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ executive_summary: reportContent.executive_summary }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Resubmit failed (${response.status})`)
      }
      const updated: ReportContent = await response.json()
      setReportContent(updated)
      setReportVersions((prev) => [updated, ...prev])
      setReportEditingSummary(false)
    } catch (err) {
      setReportError((err as Error).message)
    } finally {
      setReportActionLoading(null)
    }
  }

  async function handleLoadVersionHistory() {
    if (!result) return
    setVersionsLoading(true)
    try {
      const headers = await authHeader()
      const response = await fetch(
        `${API_BASE_URL}/report/versions?dataset_id=${encodeURIComponent(result.dataset_id)}`,
        { headers }
      )
      if (!response.ok) throw new Error('Could not load version history')
      const data = await response.json()
      setReportVersions(data.versions || [])
    } catch (err) {
      setReportError((err as Error).message)
    } finally {
      setVersionsLoading(false)
    }
  }

  async function handleDownloadReport(format: 'pdf' | 'docx') {
    if (!reportContent || reportContent.approval_status !== 'approved') return
    setReportDownloading(format)
    setReportError(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/report/${format}`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ report: reportContent, report_id: reportContent.id }),
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Download failed (${response.status})`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${reportContent.filename || 'report'}_report.${format}`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setReportError((err as Error).message)
    } finally {
      setReportDownloading(null)
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

  const [copiedChart, setCopiedChart] = useState<string | null>(null)

  function handleShareChart(chart: VizChart, key: string) {
    const summary = `${chart.title}${chart.insight ? `\n${chart.insight}` : ''}`
    navigator.clipboard
      .writeText(summary)
      .then(() => {
        setCopiedChart(key)
        setTimeout(() => setCopiedChart((prev) => (prev === key ? null : prev)), 2000)
      })
      .catch(() => {})
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
    { id: 'agent', label: 'Multi-Agent', disabled: !result },
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

            {agentResult && !agentResult.error && (
              <div className="space-y-4">
                {/* Report Agent */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-slate-100">Report Agent</p>
                      <p className="text-xs text-slate-400">
                        Turn this analysis into a shareable PDF or Word report.
                      </p>
                    </div>
                    {!reportContent && (
                      <button
                        onClick={handleGenerateReport}
                        disabled={reportGenerating}
                        className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {reportGenerating ? 'Drafting report…' : 'Generate report'}
                      </button>
                    )}
                  </div>

                  {reportError && <ErrorBanner message={reportError} />}

                  {reportContent && (
                    <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <span
                            className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${
                              reportContent.approval_status === 'approved'
                                ? 'bg-emerald-950/40 text-emerald-300'
                                : reportContent.approval_status === 'rejected'
                                  ? 'bg-red-950/40 text-red-300'
                                  : 'bg-amber-950/40 text-amber-300'
                            }`}
                          >
                            {reportContent.approval_status === 'approved'
                              ? 'Approved'
                              : reportContent.approval_status === 'rejected'
                                ? 'Rejected'
                                : 'Pending approval'}
                          </span>
                          <span className="text-xs text-slate-500">Version {reportContent.version_number}</span>
                        </div>
                      </div>

                      {/* Editable executive summary */}
                      <div className="rounded-lg bg-slate-950/60 p-3">
                        <div className="mb-1 flex items-center justify-between">
                          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                            Executive summary
                          </p>
                          <button
                            onClick={() => setReportEditingSummary((v) => !v)}
                            className="text-xs text-cyan-400 hover:text-cyan-300"
                          >
                            {reportEditingSummary ? 'Done editing' : 'Edit'}
                          </button>
                        </div>
                        {reportEditingSummary ? (
                          <textarea
                            value={reportContent.executive_summary}
                            onChange={(e) =>
                              setReportContent({
                                ...reportContent,
                                executive_summary: e.target.value,
                              })
                            }
                            rows={4}
                            className="w-full rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:border-cyan-600 focus:outline-none"
                          />
                        ) : (
                          <p className="text-sm text-slate-200">{reportContent.executive_summary}</p>
                        )}
                      </div>

                      {/* Agent discussion panel */}
                      {reportContent.agent_collaboration_narrative && (
                        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
                          <button
                            onClick={() => setShowAgentDiscussion((v) => !v)}
                            className="flex w-full items-center justify-between text-left"
                          >
                            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                              How the agents reached this conclusion
                            </p>
                            <span className="text-xs text-cyan-400">
                              {showAgentDiscussion ? 'Hide' : 'Show'}
                            </span>
                          </button>
                          {showAgentDiscussion && (
                            <div className="mt-3 space-y-3">
                              <p className="text-sm text-slate-300">
                                {reportContent.agent_collaboration_narrative}
                              </p>
                              {reportContent.agent_timeline && reportContent.agent_timeline.length > 0 && (
                                <ol className="space-y-2 border-l border-slate-800 pl-4">
                                  {reportContent.agent_timeline.map((step, i) => (
                                    <li key={i} className="relative">
                                      <span className="absolute -left-[21px] top-1 h-2 w-2 rounded-full bg-cyan-500" />
                                      <p className="text-xs font-semibold text-cyan-300">
                                        {step.agent}
                                        <span className="ml-2 font-normal text-slate-500">{step.role}</span>
                                      </p>
                                      <p className="text-xs text-slate-400">{step.detail}</p>
                                    </li>
                                  ))}
                                </ol>
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Chart previews */}
                      {reportContent.charts && reportContent.charts.length > 0 && (
                        <div>
                          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                            Charts included in this report
                          </p>
                          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                            {reportContent.charts.map((chart, i) => (
                              <div key={i} className="overflow-hidden rounded-lg border border-slate-800">
                                <img
                                  src={`data:image/png;base64,${chart.image_base64}`}
                                  alt={chart.title}
                                  className="w-full"
                                />
                                <p className="bg-slate-950/60 px-2 py-1 text-center text-[11px] text-slate-400">
                                  {chart.title}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* ---- Approval dashboard ---- */}
                      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3 space-y-3">
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                          Approval
                        </p>

                        {reportContent.approval_status === 'pending' && !showRejectForm && (
                          <div className="flex flex-wrap gap-2">
                            <button
                              onClick={handleApproveReport}
                              disabled={reportActionLoading !== null}
                              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              {reportActionLoading === 'approve' ? 'Approving…' : 'Approve'}
                            </button>
                            <button
                              onClick={() => setShowRejectForm(true)}
                              disabled={reportActionLoading !== null}
                              className="rounded-lg border border-red-800 px-4 py-2 text-sm font-medium text-red-300 hover:bg-red-950/40 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              Reject
                            </button>
                          </div>
                        )}

                        {reportContent.approval_status === 'pending' && showRejectForm && (
                          <div className="space-y-2">
                            <textarea
                              value={rejectReason}
                              onChange={(e) => setRejectReason(e.target.value)}
                              placeholder="Why is this report being rejected?"
                              rows={3}
                              className="w-full rounded-md border border-red-900 bg-slate-900 p-2 text-sm text-slate-100 focus:border-red-600 focus:outline-none"
                            />
                            <div className="flex flex-wrap gap-2">
                              <button
                                onClick={handleRejectReport}
                                disabled={reportActionLoading !== null || !rejectReason.trim()}
                                className="rounded-lg bg-red-700 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                {reportActionLoading === 'reject' ? 'Submitting…' : 'Submit rejection'}
                              </button>
                              <button
                                onClick={() => {
                                  setShowRejectForm(false)
                                  setRejectReason('')
                                }}
                                className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}

                        {reportContent.approval_status === 'rejected' && (
                          <div className="space-y-3">
                            {reportContent.rejection_reason && (
                              <p className="rounded-md border border-red-900 bg-red-950/30 p-2 text-xs text-red-200">
                                <span className="font-semibold">Reason: </span>
                                {reportContent.rejection_reason}
                              </p>
                            )}
                            <div className="flex flex-wrap gap-2">
                              <button
                                onClick={handleGenerateReport}
                                disabled={reportGenerating || reportActionLoading !== null}
                                className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                {reportGenerating ? 'Regenerating…' : 'Regenerate report'}
                              </button>
                              <button
                                onClick={handleResubmitReport}
                                disabled={reportActionLoading !== null}
                                className="rounded-lg border border-cyan-700 px-4 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-950/40 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                {reportActionLoading === 'resubmit'
                                  ? 'Resubmitting…'
                                  : 'Resubmit edited summary for approval'}
                              </button>
                            </div>
                          </div>
                        )}

                        {reportContent.approval_status === 'approved' && (
                          <div className="flex flex-wrap gap-2">
                            <button
                              onClick={() => handleDownloadReport('pdf')}
                              disabled={reportDownloading !== null}
                              className="rounded-lg border border-cyan-700 px-4 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-950/40 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              {reportDownloading === 'pdf' ? 'Downloading…' : 'Download PDF'}
                            </button>
                            <button
                              onClick={() => handleDownloadReport('docx')}
                              disabled={reportDownloading !== null}
                              className="rounded-lg border border-cyan-700 px-4 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-950/40 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              {reportDownloading === 'docx' ? 'Downloading…' : 'Download Word'}
                            </button>
                          </div>
                        )}
                      </div>

                      {/* ---- Version history ---- */}
                      <div className="rounded-lg border border-slate-800 bg-slate-950/40">
                        <button
                          onClick={() => {
                            const next = !showVersionHistory
                            setShowVersionHistory(next)
                            if (next) handleLoadVersionHistory()
                          }}
                          className="flex w-full items-center justify-between px-3 py-2 text-left"
                        >
                          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                            Version history
                          </p>
                          <span className="text-xs text-cyan-400">{showVersionHistory ? 'Hide' : 'Show'}</span>
                        </button>
                        {showVersionHistory && (
                          <div className="space-y-2 border-t border-slate-800 p-3">
                            {versionsLoading && <p className="text-xs text-slate-500">Loading…</p>}
                            {!versionsLoading && reportVersions.length === 0 && (
                              <p className="text-xs text-slate-500">No versions yet.</p>
                            )}
                            {reportVersions.map((v) => (
                              <button
                                key={v.id}
                                onClick={() => setReportContent(v)}
                                className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-xs hover:bg-slate-900 ${
                                  v.id === reportContent.id ? 'bg-slate-900' : ''
                                }`}
                              >
                                <span className="text-slate-300">Version {v.version_number}</span>
                                <span
                                  className={`rounded-full px-2 py-0.5 font-medium ${
                                    v.approval_status === 'approved'
                                      ? 'bg-emerald-950/40 text-emerald-300'
                                      : v.approval_status === 'rejected'
                                        ? 'bg-red-950/40 text-red-300'
                                        : 'bg-amber-950/40 text-amber-300'
                                  }`}
                                >
                                  {v.approval_status}
                                </span>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Key AI recommendation callout */}
                {agentResult.recommendation && (
                  <div className="rounded-xl border border-cyan-800/60 bg-cyan-950/20 p-4">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-cyan-300">
                      Key AI recommendation
                    </p>
                    <p className="text-sm text-slate-100">{agentResult.recommendation}</p>
                  </div>
                )}

                {/* Stat cards */}
                {agentResult.key_metrics && agentResult.key_metrics.length > 0 && (
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {agentResult.key_metrics.slice(0, 4).map((metric, i) => {
                      const { value, unit, label } = parseMetricCard(metric)
                      return (
                        <div
                          key={i}
                          className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-center"
                        >
                          <div className="mx-auto mb-2 flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-cyan-300">
                            {STAT_ICONS[i % STAT_ICONS.length]}
                          </div>
                          <p className="text-xl font-semibold text-white">
                            {value}
                            {unit && (
                              <span className="ml-1 text-xs font-medium text-slate-400">{unit}</span>
                            )}
                          </p>
                          <p className="mt-1 text-xs text-slate-400">{label}</p>
                        </div>
                      )
                    })}
                  </div>
                )}

                {/* AI executive summary, tabbed per specialist */}
                {agentResult.specialist_reports && agentResult.specialist_reports.length > 0 && (
                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                      AI executive summary
                    </p>
                    <div className="mb-3 flex flex-wrap gap-2 border-b border-slate-800 pb-3">
                      {agentResult.specialist_reports.map((report) => {
                        const domain = report.domain
                        const defaultDomain = agentResult.specialist_reports[0].domain
                        const isActive = (activeSpecialistTab ?? defaultDomain) === domain
                        return (
                          <button
                            key={domain}
                            onClick={() => setActiveSpecialistTab(domain)}
                            className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                              isActive
                                ? 'bg-cyan-600 text-white'
                                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            {domain} Analysis
                          </button>
                        )
                      })}
                    </div>
                    {(() => {
                      const defaultDomain = agentResult.specialist_reports[0].domain
                      const active =
                        agentResult.specialist_reports.find(
                          (r) => r.domain === (activeSpecialistTab ?? defaultDomain)
                        ) ?? agentResult.specialist_reports[0]
                      if (!active) return null
                      return active.error ? (
                        <p className="text-sm text-amber-300">{active.error}</p>
                      ) : (
                        <div className="space-y-2">
                          <p className="text-sm text-slate-200">{active.summary}</p>
                          {active.recommendation && (
                            <p className="text-sm text-indigo-300">{active.recommendation}</p>
                          )}
                        </div>
                      )
                    })()}
                  </div>
                )}

                {/* System routing & execution details (visual plan + review) */}
                {(agentResult.plan || agentResult.review || agentResult.security || agentResult.traces) && (
                  <div className="rounded-xl border border-slate-800 bg-slate-950/40">
                    <button
                      onClick={() => setShowSystemDetails((v) => !v)}
                      className="flex w-full items-center justify-between px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-200"
                    >
                      <span>System routing &amp; execution details (plan &amp; review)</span>
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        className={`h-4 w-4 transition-transform ${showSystemDetails ? 'rotate-180' : ''}`}
                      >
                        <path
                          d="M6 9l6 6 6-6"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>

                    {showSystemDetails && (
                      <div className="space-y-4 border-t border-slate-800 p-4">
                        <div className="grid gap-4 lg:grid-cols-2">
                          {/* Panel A: routing flow, icon-first, minimal text */}
                          {agentResult.plan && agentResult.plan.length > 0 && (
                            <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
                              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-cyan-300">
                                Routing plan
                              </p>
                              <div className="flex flex-wrap items-center gap-2">
                                <div className="flex flex-col items-center gap-1">
                                  <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-800 text-cyan-300">
                                    <DatasetIcon />
                                  </div>
                                  <span className="text-[10px] text-slate-500">Source</span>
                                </div>
                                {agentResult.plan.map((step) => (
                                  <div key={step.step} className="flex items-center gap-2">
                                    <ArrowIcon />
                                    <div className="flex flex-col items-center gap-1">
                                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-800 text-cyan-300">
                                        {domainIcon(step.agent)}
                                      </div>
                                      <span className="text-[10px] font-medium text-slate-300">
                                        {step.agent}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                                <ArrowIcon />
                                <div className="flex flex-col items-center gap-1">
                                  <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-800 text-indigo-300">
                                    <OrchestratorIcon />
                                  </div>
                                  <span className="text-[10px] text-slate-500">Manager</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Panel B: validation, badge chips instead of sentences */}
                          {(agentResult.review || agentResult.security) && (
                            <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
                              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-emerald-300">
                                Validation
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {agentResult.review && (
                                  <span
                                    className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                                      agentResult.review.approved
                                        ? 'bg-emerald-900/50 text-emerald-300'
                                        : 'bg-amber-900/50 text-amber-300'
                                    }`}
                                  >
                                    <span
                                      className={`h-1.5 w-1.5 rounded-full ${
                                        agentResult.review.approved ? 'bg-emerald-400' : 'bg-amber-400'
                                      }`}
                                    />
                                    Quality: <span className="capitalize">{agentResult.review.overall_quality}</span>
                                  </span>
                                )}
                                {agentResult.security && (
                                  <span
                                    className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                                      agentResult.security.safe_to_show
                                        ? 'bg-emerald-900/50 text-emerald-300'
                                        : 'bg-amber-900/50 text-amber-300'
                                    }`}
                                  >
                                    <span
                                      className={`h-1.5 w-1.5 rounded-full ${
                                        agentResult.security.safe_to_show ? 'bg-emerald-400' : 'bg-amber-400'
                                      }`}
                                    />
                                    Risk: <span className="capitalize">{agentResult.security.risk_level}</span>
                                  </span>
                                )}
                                {!agentResult.review?.issues?.length && !agentResult.security?.findings?.length && (
                                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-900/50 px-2.5 py-1 text-[11px] font-medium text-emerald-300">
                                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                                    No issues found
                                  </span>
                                )}
                              </div>
                              {(agentResult.review?.issues?.length || agentResult.security?.findings?.length) ? (
                                <ul className="mt-3 space-y-1.5 text-[11px] text-amber-300">
                                  {agentResult.review?.issues?.map((issue, i) => (
                                    <li key={`ri-${i}`} className="flex items-start gap-1.5">
                                      <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-amber-400" />
                                      {issue}
                                    </li>
                                  ))}
                                  {agentResult.security?.findings?.map((f, i) => (
                                    <li key={`sf-${i}`} className="flex items-start gap-1.5">
                                      <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-amber-400" />
                                      {f}
                                    </li>
                                  ))}
                                </ul>
                              ) : null}
                            </div>
                          )}
                        </div>

                        {agentResult.traces && agentResult.traces.length > 0 && (
                          <div>
                            <p className="mb-3 text-sm font-medium text-slate-300">Agent traces</p>
                            <div className="max-h-64 overflow-y-auto pr-1">
                              {agentResult.traces.map((t, i) => {
                                const isLast = i === agentResult.traces!.length - 1
                                const dotColor = t.error
                                  ? 'bg-red-400'
                                  : t.action === 'completed' || t.action === 'finished'
                                    ? 'bg-emerald-400'
                                    : 'bg-cyan-400'
                                const actionLabel = t.action.replace(/_/g, ' ')
                                const time = new Date(t.timestamp * 1000).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit',
                                  second: '2-digit',
                                })
                                return (
                                  <div key={i} className="relative flex gap-3 pb-4 last:pb-0">
                                    {!isLast && (
                                      <span className="absolute left-[5px] top-3 h-full w-px bg-slate-800" />
                                    )}
                                    <span
                                      className={`relative z-10 mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${dotColor}`}
                                    />
                                    <div className="flex flex-1 items-center justify-between gap-2 text-xs">
                                      <span>
                                        <span className="font-medium text-cyan-300">{t.agent}</span>
                                        <span className="ml-1.5 capitalize text-slate-400">{actionLabel}</span>
                                        {t.error && <span className="ml-1.5 text-red-400">— {t.error}</span>}
                                      </span>
                                      <span className="shrink-0 text-[10px] text-slate-600">{time}</span>
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        )}

                        <button
                          onClick={() => setShowSystemDetails(false)}
                          className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-800 bg-slate-900/60 py-2 text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-200"
                        >
                          Close system details (plan &amp; review)
                          <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4 rotate-180">
                            <path
                              d="M6 9l6 6 6-6"
                              stroke="currentColor"
                              strokeWidth="1.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---------- Multi-Agent tab (live orchestration) ---------- */}
      {result && activeTab === 'agent' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="text-sm font-semibold text-slate-100">Multi-Agent Analysis</p>
            <p className="mb-3 text-xs text-slate-400">
              Ask a question in plain English. The Manager Agent plans the work, delegates
              to specialists, and shows its progress live as it happens.
            </p>
            <textarea
              value={agentQuery}
              onChange={(e) => setAgentQuery(e.target.value)}
              rows={2}
              placeholder="e.g. Why did revenue drop, and what should we do about it?"
              className="w-full rounded-md border border-slate-700 bg-slate-950 p-2 text-sm text-slate-100 focus:border-cyan-600 focus:outline-none"
            />
            <button
              onClick={handleRunAgentStream}
              disabled={agentStreamRunning || !agentQuery.trim()}
              className="mt-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {agentStreamRunning ? 'Running…' : 'Run'}
            </button>
          </div>

          {agentStreamError && <ErrorBanner message={agentStreamError} />}

          {agentStreamEvents.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Live progress
              </p>
              <ol className="space-y-2">
                {agentStreamEvents.map((event, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="mt-0.5 w-4 shrink-0 text-center">
                      {agentEventIcon(event.type)}
                    </span>
                    <div>
                      <span className="font-medium text-slate-200">{event.agent}</span>
                      <span className="ml-2 text-slate-400">{event.message}</span>
                    </div>
                  </li>
                ))}
                {agentStreamRunning && (
                  <li className="flex items-center gap-2 text-sm text-slate-500">
                    <span className="w-4 text-center">⟳</span>
                    <span>working…</span>
                  </li>
                )}
              </ol>
            </div>
          )}

          {agentStreamResult && (
            <div className="rounded-xl border border-cyan-800 bg-cyan-950/20 p-4">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-cyan-400">
                Final Answer
              </p>
              {agentStreamResult.result?.summary && (
                <p className="text-sm text-slate-100">{agentStreamResult.result.summary}</p>
              )}
              {agentStreamResult.result?.recommendation && (
                <p className="mt-2 text-sm text-slate-300">
                  <span className="font-medium text-slate-200">Recommendation: </span>
                  {agentStreamResult.result.recommendation}
                </p>
              )}
              {agentStreamResult.result?.key_metrics && agentStreamResult.result.key_metrics.length > 0 && (
                <ul className="mt-2 list-inside list-disc text-sm text-slate-300">
                  {agentStreamResult.result.key_metrics.map((m, i) => (
                    <li key={i}>{m}</li>
                  ))}
                </ul>
              )}
              <p className="mt-3 text-xs text-slate-500">
                Status: {agentStreamResult.status}
                {agentStreamResult.result?.participating_agents && agentStreamResult.result.participating_agents.length > 0 &&
                  ` · Agents: ${agentStreamResult.result.participating_agents.join(', ')}`}
              </p>
            </div>
          )}

          {agentApproval && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
              <div className="mb-2 flex items-center justify-between gap-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Human Approval
                </p>
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                    agentApproval.approval_status === 'approved'
                      ? 'bg-emerald-900/40 text-emerald-300'
                      : agentApproval.approval_status === 'rejected'
                        ? 'bg-red-900/40 text-red-300'
                        : 'bg-amber-900/40 text-amber-300'
                  }`}
                >
                  {agentApproval.approval_status === 'approved'
                    ? 'Approved'
                    : agentApproval.approval_status === 'rejected'
                      ? 'Rejected'
                      : 'Pending approval'}
                </span>
              </div>

              {agentApprovalError && <ErrorBanner message={agentApprovalError} />}

              {agentApproval.approval_status === 'rejected' && agentApproval.rejection_reason && (
                <p className="mb-2 text-xs text-red-300">Reason: {agentApproval.rejection_reason}</p>
              )}

              {agentApproval.approval_status === 'pending' && !agentShowRejectForm && (
                <div className="flex gap-2">
                  <button
                    onClick={handleApproveAgentWorkflow}
                    disabled={agentApprovalActionLoading !== null}
                    className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {agentApprovalActionLoading === 'approve' ? 'Approving…' : 'Approve'}
                  </button>
                  <button
                    onClick={() => setAgentShowRejectForm(true)}
                    disabled={agentApprovalActionLoading !== null}
                    className="rounded-lg border border-red-800 px-3 py-1.5 text-xs font-medium text-red-300 hover:bg-red-950/40 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Reject
                  </button>
                </div>
              )}

              {agentApproval.approval_status === 'pending' && agentShowRejectForm && (
                <div className="space-y-2">
                  <textarea
                    value={agentRejectReason}
                    onChange={(e) => setAgentRejectReason(e.target.value)}
                    rows={2}
                    placeholder="Why is this being rejected?"
                    className="w-full rounded-md border border-slate-700 bg-slate-950 p-2 text-sm text-slate-100 focus:border-red-600 focus:outline-none"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={handleRejectAgentWorkflow}
                      disabled={agentApprovalActionLoading !== null || !agentRejectReason.trim()}
                      className="rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {agentApprovalActionLoading === 'reject' ? 'Rejecting…' : 'Confirm reject'}
                    </button>
                    <button
                      onClick={() => { setAgentShowRejectForm(false); setAgentRejectReason('') }}
                      className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ---------- Run history ---------- */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <button
              onClick={handleToggleAgentHistory}
              className="flex w-full items-center justify-between text-left"
            >
              <p className="text-sm font-semibold text-slate-100">Run History</p>
              <span className="text-xs text-slate-400">{agentHistoryOpen ? 'Hide' : 'Show'}</span>
            </button>

            {agentHistoryOpen && (
              <div className="mt-3 space-y-3">
                {agentHistoryError && <ErrorBanner message={agentHistoryError} />}
                {agentHistoryLoading && <p className="text-xs text-slate-500">Loading…</p>}
                {!agentHistoryLoading && agentHistoryRuns.length === 0 && (
                  <p className="text-xs text-slate-500">No past runs for this dataset yet.</p>
                )}
                {agentHistoryRuns.length > 0 && (
                  <ul className="space-y-1">
                    {agentHistoryRuns.map((run) => (
                      <li key={run.workflow_id}>
                        <button
                          onClick={() => handleOpenHistoryRun(run.workflow_id)}
                          disabled={historyLoadingId === run.workflow_id}
                          className="w-full rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-left text-xs hover:border-cyan-700 disabled:opacity-60"
                        >
                          <p className="font-medium text-slate-200">{run.user_query || run.goal || 'Untitled run'}</p>
                          <p className="text-slate-500">
                            {new Date(run.created_at).toLocaleString()} · {run.status}
                          </p>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}

                {selectedHistoryRun && (
                  <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-4">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                      {selectedHistoryRun.user_query}
                    </p>
                    {selectedHistoryRun.state.final_output?.summary && (
                      <p className="text-sm text-slate-100">{selectedHistoryRun.state.final_output.summary}</p>
                    )}
                    {selectedHistoryRun.state.final_output?.recommendation && (
                      <p className="mt-2 text-sm text-slate-300">
                        <span className="font-medium text-slate-200">Recommendation: </span>
                        {selectedHistoryRun.state.final_output.recommendation}
                      </p>
                    )}
                    <p className="mt-2 text-xs text-slate-500">
                      {new Date(selectedHistoryRun.created_at).toLocaleString()} · {selectedHistoryRun.status}
                    </p>

                    {selectedHistoryApproval && (
                      <div className="mt-3 border-t border-slate-800 pt-3">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                            Human Approval
                          </p>
                          <span
                            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                              selectedHistoryApproval.approval_status === 'approved'
                                ? 'bg-emerald-900/40 text-emerald-300'
                                : selectedHistoryApproval.approval_status === 'rejected'
                                  ? 'bg-red-900/40 text-red-300'
                                  : 'bg-amber-900/40 text-amber-300'
                            }`}
                          >
                            {selectedHistoryApproval.approval_status === 'approved'
                              ? 'Approved'
                              : selectedHistoryApproval.approval_status === 'rejected'
                                ? 'Rejected'
                                : 'Pending approval'}
                          </span>
                        </div>

                        {selectedHistoryApproval.approval_status === 'rejected' && selectedHistoryApproval.rejection_reason && (
                          <p className="mb-2 text-xs text-red-300">Reason: {selectedHistoryApproval.rejection_reason}</p>
                        )}

                        {selectedHistoryApproval.approval_status === 'pending' && !historyShowRejectForm && (
                          <div className="flex gap-2">
                            <button
                              onClick={handleApproveHistoryRun}
                              disabled={historyApprovalActionLoading !== null}
                              className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              {historyApprovalActionLoading === 'approve' ? 'Approving…' : 'Approve'}
                            </button>
                            <button
                              onClick={() => setHistoryShowRejectForm(true)}
                              disabled={historyApprovalActionLoading !== null}
                              className="rounded-lg border border-red-800 px-3 py-1.5 text-xs font-medium text-red-300 hover:bg-red-950/40 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              Reject
                            </button>
                          </div>
                        )}

                        {selectedHistoryApproval.approval_status === 'pending' && historyShowRejectForm && (
                          <div className="space-y-2">
                            <textarea
                              value={historyRejectReason}
                              onChange={(e) => setHistoryRejectReason(e.target.value)}
                              rows={2}
                              placeholder="Why is this being rejected?"
                              className="w-full rounded-md border border-slate-700 bg-slate-950 p-2 text-sm text-slate-100 focus:border-red-600 focus:outline-none"
                            />
                            <div className="flex gap-2">
                              <button
                                onClick={handleRejectHistoryRun}
                                disabled={historyApprovalActionLoading !== null || !historyRejectReason.trim()}
                                className="rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                {historyApprovalActionLoading === 'reject' ? 'Rejecting…' : 'Confirm reject'}
                              </button>
                              <button
                                onClick={() => { setHistoryShowRejectForm(false); setHistoryRejectReason('') }}
                                className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
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
                  <option value="area">Area</option>
                  <option value="scatter">Scatter</option>
                  <option value="pie">Pie</option>
                  <option value="box">Box plot</option>
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
              {(vizChartType === 'bar' || vizChartType === 'line' || vizChartType === 'scatter' || vizChartType === 'area' || vizChartType === 'pie' || vizChartType === 'box') && (
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
            <div className="space-y-4">
              {vizResult.charts.length > 1 && (
                <p className="text-xs text-purple-300">
                  Generated {vizResult.charts.length} charts for this request.
                </p>
              )}
              {vizResult.charts.map((chart, i) => (
                <div key={i} className="rounded-xl border border-slate-800 bg-slate-950/30 p-4 space-y-2">
                  {chart.reasoning && <p className="text-xs text-purple-300">AI interpretation: {chart.reasoning}</p>}
                  <Plot
                    data={chart.figure.data}
                    layout={{ ...chart.figure.layout, autosize: true }}
                    config={{
                      responsive: true,
                      displaylogo: false,
                      displayModeBar: true,
                      toImageButtonOptions: { format: 'png', filename: chart.title || 'chart' },
                    }}
                    useResizeHandler
                    style={{ width: '100%', height: '420px' }}
                    className="rounded-lg border border-slate-800"
                  />
                  {chart.insight && <p className="text-sm text-slate-300">{chart.insight}</p>}
                  <button
                    onClick={() => handleShareChart(chart, `result-${i}`)}
                    className="text-xs text-slate-400 hover:text-slate-200"
                  >
                    {copiedChart === `result-${i}` ? 'Copied!' : 'Copy summary to share'}
                  </button>
                </div>
              ))}
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
                  <div key={i} className="rounded-lg border border-slate-800 overflow-hidden p-2 space-y-2">
                    <Plot
                      data={chart.figure.data}
                      layout={{ ...chart.figure.layout, autosize: true }}
                      config={{
                        responsive: true,
                        displaylogo: false,
                        displayModeBar: true,
                        toImageButtonOptions: { format: 'png', filename: chart.title || 'chart' },
                      }}
                      useResizeHandler
                      style={{ width: '100%', height: '340px' }}
                    />
                    {chart.insight && <p className="text-xs text-slate-400 px-1">{chart.insight}</p>}
                    <button
                      onClick={() => handleShareChart(chart, `dash-${i}`)}
                      className="text-xs text-slate-500 hover:text-slate-300 px-1"
                    >
                      {copiedChart === `dash-${i}` ? 'Copied!' : 'Copy summary to share'}
                    </button>
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

function agentEventIcon(type: string): string {
  switch (type) {
    case 'manager_start':
    case 'plan_created':
    case 'plan_ordered':
      return '🧭'
    case 'wave_started':
      return '▶'
    case 'task_started':
      return '⟳'
    case 'task_completed':
      return '✓'
    case 'task_failed':
      return '✗'
    case 'task_skipped':
      return '⊘'
    case 'task_retrying':
      return '↻'
    case 'reviewer_check_start':
    case 'reviewer_check_completed':
      return '📝'
    case 'reviewer_check_failed':
      return '⚠'
    case 'security_check_start':
    case 'security_check_completed':
      return '🔒'
    case 'security_check_failed':
      return '⚠'
    case 'quality_check_start':
    case 'quality_check_completed':
      return '🛡'
    case 'quality_check_failed':
      return '⚠'
    case 'finished':
      return '🏁'
    default:
      return '•'
  }
}

function Stat({ label, value, isText }: { label: string; value: number | string; isText?: boolean }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <p className="text-xs text-slate-400">{label}</p>
      <p className={`font-semibold ${isText ? 'text-lg capitalize' : 'text-2xl'}`}>{value}</p>
    </div>
  )
}