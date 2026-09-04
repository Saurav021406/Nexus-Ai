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
  kind?: 'tabular' | 'document'
  // Tabular fields - present when kind is 'tabular' (or omitted, for
  // backward compatibility with responses from before documents existed)
  row_count?: number
  column_count?: number
  columns?: ColumnInfo[]
  preview_rows?: Record<string, string>[]
  // Document fields - present when kind is 'document' (PDF/Word upload)
  document_type?: 'pdf' | 'docx'
  page_count?: number | null
  paragraph_count?: number | null
  word_count?: number
  char_count?: number
  text_preview?: string
  extracted_text?: string
  chunks_ingested?: number | null
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
    specialist_reports?: SpecialistReport[]
    data_summary?: string
  }
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

interface UsageStats {
  since: number
  counters: {
    cache_hits: number
    cache_misses: number
    domain_gate_rejections: number
    evidence_gate_rejections: number
    consensus_calls_fast: number
    consensus_calls_full: number
  }
  cache_hit_rate: number | null
  total_calls_avoided: number
  estimated_tokens_saved: {
    from_avoided_calls: number
    from_fast_tier_usage: number
    total: number
    methodology: string
  }
}

interface HistoryItem {
  id: string
  filename: string
  row_count: number
  column_count: number
  created_at: string
}

interface ChatSource {
  excerpt_number: number
  chunk_index: number | null
  preview: string
  text?: string
  score?: number | null
}

interface ChatConsensus {
  models_used: string[]
  ranking: string[]
  agreement_score: number
  confidence: number
  contradictions: string[]
  synthesis_notes: string[]
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[] | null
  consensus?: ChatConsensus | null
}

interface AutoMLModelResult {
  name: string
  cv_score_mean: number
  cv_score_std: number
  cv_metric: string
  test_metrics: Record<string, number | string>
}

interface AutoMLShapFeature {
  feature: string
  importance: number
}

interface AutoMLBusinessSummary {
  summary: string
  key_metrics: string[]
  recommendation: string
}

interface AutoMLClassImbalance {
  detected: boolean
  ratio: number
  class_counts: Record<string, number>
}

interface AutoMLResult {
  problem_type: 'classification' | 'regression'
  target_column: string
  feature_columns: string[]
  n_rows_used: number
  n_rows_dropped: number
  primary_metric: string
  best_model_name: string
  model_id: string | null
  models: AutoMLModelResult[]
  shap_importances: AutoMLShapFeature[]
  shap_unavailable_reason: string | null
  warnings: string[]
  excluded_id_columns: string[]
  class_imbalance: AutoMLClassImbalance | null
  business_summary: AutoMLBusinessSummary
  version_id: string
  version_number: number
}

interface AutoMLVersion {
  id: string
  version_number: number
  created_at: string
  problem_type: string
  target_column: string
  best_model_name: string
  models: AutoMLModelResult[]
}

interface AutoMLPredictResponse {
  predictions: (string | number)[]
  model_id: string
  n_rows: number
  probabilities?: Record<string, number>[]
  explanations?: PredictionExplanation[]
}

interface AutoMLClusterResult {
  n_clusters: number
  silhouette_score: number
  cluster_sizes: Record<string, number>
  cluster_profiles: Record<string, Record<string, number>>
  feature_columns: string[]
}

interface AnomalyRecord {
  row_index: number
  anomaly_score: number
  values: Record<string, number>
}

interface AnomalyResult {
  n_rows_analyzed: number
  n_anomalies: number
  anomaly_rate: number
  feature_columns: string[]
  anomalies: AnomalyRecord[]
  anomalies_truncated: boolean
}

interface PredictionExplanationItem {
  feature: string
  impact: number
}

interface PredictionExplanation {
  explanation: PredictionExplanationItem[]
  unavailable_reason: string | null
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
  version_id: string
  version_number: number
}

interface CleaningVersion {
  id: string
  version_number: number
  created_at: string
  report: CleanReport
  options: Record<string, unknown>
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

export type WorkspaceTab = 'upload' | 'profile' | 'analysis' | 'agent' | 'chat' | 'forecast' | 'automl' | 'clean' | 'eda' | 'visualize' | 'history' | 'usage'

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
  const [showFullDataSummary, setShowFullDataSummary] = useState(false)
  const [expandedReasoning, setExpandedReasoning] = useState<Set<number>>(new Set())
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
  const [activeSource, setActiveSource] = useState<ChatSource | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  const [forecastColumns, setForecastColumns] = useState<ForecastColumnsInfo | null>(null)
  const [forecastColumnsError, setForecastColumnsError] = useState<string | null>(null)
  const [automlTargetColumn, setAutomlTargetColumn] = useState('')
  const [automlProblemType, setAutomlProblemType] = useState<'auto' | 'classification' | 'regression'>('auto')
  const [automlRunning, setAutomlRunning] = useState(false)
  const [automlError, setAutomlError] = useState<string | null>(null)
  const [automlResult, setAutomlResult] = useState<AutoMLResult | null>(null)
  const [automlMode, setAutomlMode] = useState<'train' | 'cluster' | 'anomaly'>('train')
  const [automlPredictInputs, setAutomlPredictInputs] = useState<Record<string, string>>({})
  const [automlPredicting, setAutomlPredicting] = useState(false)
  const [automlPredictError, setAutomlPredictError] = useState<string | null>(null)
  const [automlPredictResult, setAutomlPredictResult] = useState<AutoMLPredictResponse | null>(null)
  const [automlExplainPrediction, setAutomlExplainPrediction] = useState(false)
  const [batchPredictFile, setBatchPredictFile] = useState<File | null>(null)
  const [batchPredicting, setBatchPredicting] = useState(false)
  const [batchPredictError, setBatchPredictError] = useState<string | null>(null)
  const [clusterFeatureColumns, setClusterFeatureColumns] = useState<string[]>([])
  const [clustering, setClustering] = useState(false)
  const [clusterError, setClusterError] = useState<string | null>(null)
  const [clusterResult, setClusterResult] = useState<AutoMLClusterResult | null>(null)
  const [anomalyFeatureColumns, setAnomalyFeatureColumns] = useState<string[]>([])
  const [anomalyContamination, setAnomalyContamination] = useState(0.05)
  const [detectingAnomalies, setDetectingAnomalies] = useState(false)
  const [anomalyError, setAnomalyError] = useState<string | null>(null)
  const [anomalyResult, setAnomalyResult] = useState<AnomalyResult | null>(null)
  const [automlVersions, setAutomlVersions] = useState<AutoMLVersion[]>([])
  const [automlVersionsLoading, setAutomlVersionsLoading] = useState(false)
  const [automlVersionsError, setAutomlVersionsError] = useState<string | null>(null)
  const [selectedTarget, setSelectedTarget] = useState<string>('')
  const [forecasting, setForecasting] = useState(false)
  const [forecastError, setForecastError] = useState<string | null>(null)
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null)

  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null)
  const [usageLoading, setUsageLoading] = useState(false)
  const [usageError, setUsageError] = useState<string | null>(null)
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
  const [cleanDownloading, setCleanDownloading] = useState(false)
  const [cleanDownloadError, setCleanDownloadError] = useState<string | null>(null)
  const [cleaningVersions, setCleaningVersions] = useState<CleaningVersion[]>([])
  const [cleaningVersionsLoading, setCleaningVersionsLoading] = useState(false)
  const [cleaningVersionsError, setCleaningVersionsError] = useState<string | null>(null)

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
    setExpandedReasoning(new Set())
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

  async function loadUsageStats() {
    setUsageLoading(true)
    setUsageError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/stats/usage`, { headers })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load usage stats (${response.status})`)
      }
      setUsageStats(await response.json())
    } catch (err) {
      setUsageError((err as Error).message)
    } finally {
      setUsageLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'usage') {
      loadUsageStats()
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

  function handleExportAgentTranscript() {
    if (agentStreamEvents.length === 0 && !agentStreamResult) return

    const lines: string[] = [
      `# Multi-Agent run transcript${result?.filename ? ` - ${result.filename}` : ''}`,
      `Exported ${new Date().toLocaleString()}`,
      '',
      `**Question:** ${agentQuery}`,
      '',
      '## Live progress',
      '',
    ]

    for (const event of agentStreamEvents) {
      lines.push(`- **${event.agent}** - ${event.message}`)
    }

    if (agentStreamResult) {
      lines.push('')
      lines.push('## Final answer')
      lines.push('')
      lines.push(`Status: ${agentStreamResult.status}`)
      if (agentStreamResult.result?.summary) {
        lines.push('')
        lines.push(agentStreamResult.result.summary)
      }
      if (agentStreamResult.result?.key_metrics && agentStreamResult.result.key_metrics.length > 0) {
        lines.push('')
        for (const m of agentStreamResult.result.key_metrics) {
          lines.push(`- ${m}`)
        }
      }
      if (agentStreamResult.result?.recommendation) {
        lines.push('')
        lines.push(`**Recommendation:** ${agentStreamResult.result.recommendation}`)
      }
    }

    const baseName = result?.filename?.includes('.') ? result.filename.split('.').slice(0, -1).join('.') : 'agent'
    downloadTextFile(`${baseName}_agent_transcript.md`, lines.join('\n'))
  }

  async function handleRunAgentStream() {
    if (!result || !agentQuery.trim()) return
    setAgentStreamRunning(true)
    setAgentStreamEvents([])
    setAgentStreamResult(null)
    setShowFullDataSummary(false)
    setAgentStreamError(null)

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

  function handleExportChatTranscript() {
    if (messages.length === 0) return

    const lines: string[] = [
      `# Chat transcript${result?.filename ? ` - ${result.filename}` : ''}`,
      `Exported ${new Date().toLocaleString()}`,
      '',
    ]

    for (const msg of messages) {
      lines.push(`**${msg.role === 'user' ? 'You' : 'Assistant'}:** ${msg.content}`)
      if (msg.sources && msg.sources.length > 0) {
        lines.push('')
        lines.push('Sources:')
        for (const s of msg.sources) {
          lines.push(`- [Excerpt ${s.excerpt_number}] ${s.preview}`)
        }
      }
      if (msg.consensus) {
        lines.push('')
        lines.push(
          `_Confidence: ${Math.round(msg.consensus.confidence * 100)}% - Models: ${msg.consensus.models_used.join(', ')}_`
        )
      }
      lines.push('')
    }

    const baseName = result?.filename?.includes('.') ? result.filename.split('.').slice(0, -1).join('.') : 'chat'
    downloadTextFile(`${baseName}_chat_transcript.md`, lines.join('\n'))
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
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, sources: data.sources ?? null, consensus: data.consensus ?? null },
      ])
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

  async function handleRunAutoML() {
    if (!result || !automlTargetColumn) return
    setAutomlRunning(true)
    setAutomlError(null)
    setAutomlResult(null)
    setAutomlPredictInputs({})
    setAutomlPredictResult(null)
    setAutomlPredictError(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/automl/run`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: result.dataset_id,
          target_column: automlTargetColumn,
          problem_type: automlProblemType === 'auto' ? null : automlProblemType,
        }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `AutoML run failed (${response.status})`)
      }
      setAutomlResult(await response.json())
      loadAutomlVersions()
    } catch (err) {
      setAutomlError((err as Error).message)
    } finally {
      setAutomlRunning(false)
    }
  }

  async function loadAutomlVersions() {
    if (!result) return
    setAutomlVersionsLoading(true)
    setAutomlVersionsError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(
        `${API_BASE_URL}/automl/versions?dataset_id=${encodeURIComponent(result.dataset_id)}`,
        { headers }
      )
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load AutoML history (${response.status})`)
      }
      const body = await response.json()
      setAutomlVersions(body.versions)
    } catch (err) {
      setAutomlVersionsError((err as Error).message)
    } finally {
      setAutomlVersionsLoading(false)
    }
  }

  async function handleAutoMLPredict() {
    if (!automlResult?.model_id) return
    setAutomlPredicting(true)
    setAutomlPredictError(null)
    setAutomlPredictResult(null)

    try {
      const headers = await authHeader()
      // Numbers are sent as numbers (not strings) so the backend's numeric
      // imputer/scaler treat them correctly - a blank field is simply
      // omitted so the trained imputer fills it in, same as a missing
      // value anywhere else in this app.
      const row: Record<string, string | number> = {}
      for (const col of automlResult.feature_columns) {
        const raw = automlPredictInputs[col]
        if (raw === undefined || raw.trim() === '') continue
        const asNumber = Number(raw)
        row[col] = Number.isNaN(asNumber) ? raw : asNumber
      }

      const response = await fetch(`${API_BASE_URL}/automl/predict`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: automlResult.model_id, rows: [row], explain: automlExplainPrediction }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Prediction failed (${response.status})`)
      }
      setAutomlPredictResult(await response.json())
    } catch (err) {
      setAutomlPredictError((err as Error).message)
    } finally {
      setAutomlPredicting(false)
    }
  }

  async function handleBatchPredict() {
    if (!automlResult?.model_id || !batchPredictFile) return
    setBatchPredicting(true)
    setBatchPredictError(null)

    try {
      const headers = await authHeader()
      const formData = new FormData()
      formData.append('model_id', automlResult.model_id)
      formData.append('file', batchPredictFile)

      const response = await fetch(`${API_BASE_URL}/automl/predict/csv`, {
        method: 'POST',
        headers, // no Content-Type here - the browser sets the correct multipart boundary for FormData
        body: formData,
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Batch prediction failed (${response.status})`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const disposition = response.headers.get('content-disposition') || ''
      const match = disposition.match(/filename="(.+)"/)
      a.download = match ? match[1] : 'predictions.csv'
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setBatchPredictError((err as Error).message)
    } finally {
      setBatchPredicting(false)
    }
  }

  async function handleDetectAnomalies() {
    if (!result) return
    setDetectingAnomalies(true)
    setAnomalyError(null)
    setAnomalyResult(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/automl/anomalies`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: result.dataset_id,
          feature_columns: anomalyFeatureColumns.length > 0 ? anomalyFeatureColumns : null,
          contamination: anomalyContamination,
        }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Anomaly detection failed (${response.status})`)
      }
      setAnomalyResult(await response.json())
    } catch (err) {
      setAnomalyError((err as Error).message)
    } finally {
      setDetectingAnomalies(false)
    }
  }

  async function handleRunClustering() {
    if (!result) return
    setClustering(true)
    setClusterError(null)
    setClusterResult(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/automl/cluster`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: result.dataset_id,
          feature_columns: clusterFeatureColumns.length > 0 ? clusterFeatureColumns : null,
        }),
      })
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Clustering failed (${response.status})`)
      }
      setClusterResult(await response.json())
    } catch (err) {
      setClusterError((err as Error).message)
    } finally {
      setClustering(false)
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

  useEffect(() => {
    if (activeTab === 'clean' && result) {
      loadCleaningVersions()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, result])

  useEffect(() => {
    if (activeTab === 'automl' && result) {
      loadAutomlVersions()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, result])

  async function handleApplyCleaning() {
    if (!result) return
    setCleaning(true)
    setCleanError(null)
    setCleanResult(null)
    setCleanDownloadError(null)

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
      loadCleaningVersions()
    } catch (err) {
      setCleanError((err as Error).message)
    } finally {
      setCleaning(false)
    }
  }

  async function handleDownloadCleaned(versionId?: string, versionNumber?: number) {
    if (!result) return
    setCleanDownloading(true)
    setCleanDownloadError(null)

    try {
      const headers = await authHeader()
      const response = await fetch(`${API_BASE_URL}/clean/download`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: result.dataset_id, version_id: versionId ?? null }),
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Download failed (${response.status})`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const baseName = result.filename?.includes('.') ? result.filename.split('.').slice(0, -1).join('.') : result.filename || 'dataset'
      const suffix = versionNumber ? `_cleaned_v${versionNumber}` : '_cleaned'
      a.download = `${baseName}${suffix}.csv`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setCleanDownloadError((err as Error).message)
    } finally {
      setCleanDownloading(false)
    }
  }

  async function loadCleaningVersions() {
    if (!result) return
    setCleaningVersionsLoading(true)
    setCleaningVersionsError(null)
    try {
      const headers = await authHeader()
      const response = await fetch(
        `${API_BASE_URL}/clean/versions?dataset_id=${encodeURIComponent(result.dataset_id)}`,
        { headers }
      )
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Could not load cleaning history (${response.status})`)
      }
      const body = await response.json()
      setCleaningVersions(body.versions)
    } catch (err) {
      setCleaningVersionsError((err as Error).message)
    } finally {
      setCleaningVersionsLoading(false)
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
    { id: 'analysis', label: 'AI analysis', disabled: !result || result?.kind === 'document' },
    { id: 'agent', label: 'Multi-Agent', disabled: !result },
    { id: 'chat', label: 'Ask your data', disabled: !result },
    { id: 'forecast', label: 'Forecast', disabled: !result || result?.kind === 'document' },
    { id: 'automl', label: 'AutoML', disabled: !result || result?.kind === 'document' },
    { id: 'clean', label: 'Clean Data', disabled: !result || result?.kind === 'document' },
    { id: 'eda', label: 'EDA & Charts', disabled: !result || result?.kind === 'document' },
    { id: 'visualize', label: 'Visualize', disabled: !result || result?.kind === 'document' },
    { id: 'history', label: 'History' },
    { id: 'usage', label: 'Usage' },
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
            <p className="mt-1 text-sm text-slate-400">Choose a CSV, Excel (.xlsx), PDF, or Word (.docx) file.</p>
          </div>
          <label className="block rounded-xl border border-dashed border-slate-700 bg-slate-950/30 p-6 transition hover:border-blue-500/70">
            <span className="sr-only">Choose CSV file</span>
            <input
              type="file"
              accept=".csv,.xlsx,.xls,.pdf,.docx"
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
      {result && activeTab === 'profile' && result.kind === 'document' && (
        <div className="space-y-6 py-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Type" value={(result.document_type || 'document').toUpperCase()} isText />
            {result.page_count != null && <Stat label="Pages" value={result.page_count} />}
            {result.paragraph_count != null && <Stat label="Paragraphs" value={result.paragraph_count} />}
            <Stat label="Words" value={result.word_count ?? 0} />
            <Stat
              label="Chunks indexed"
              value={result.chunks_ingested != null ? result.chunks_ingested : 'pending'}
              isText={result.chunks_ingested == null}
            />
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Extracted text preview
            </p>
            <p className="whitespace-pre-wrap text-sm text-slate-300">
              {result.text_preview || 'No preview available.'}
            </p>
          </div>

          <div className="rounded-lg border border-amber-800 bg-amber-950/20 px-3 py-2 text-sm text-amber-300">
            This is a document dataset. AI analysis, Multi-Agent, Chat, Forecast, Clean Data, EDA, and
            Visualize are built for spreadsheet data (CSV/Excel) and aren't available for documents yet -
            that's coming with the RAG-based document Q&amp;A feature.
          </div>
        </div>
      )}

      {result && activeTab === 'profile' && result.kind !== 'document' && (
        <div className="space-y-6 py-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Rows" value={result.row_count ?? 0} />
            <Stat label="Columns" value={result.column_count ?? 0} />
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
                {(result.columns ?? []).map((col) => (
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
                    {(result.columns ?? []).map((col) => (
                      <th key={col.name} className="text-left p-2 whitespace-nowrap">
                        {col.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(result.preview_rows ?? []).map((row, i) => (
                    <tr key={i} className="border-t border-slate-800">
                      {(result.columns ?? []).map((col) => (
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
      {result && result.kind !== 'document' && activeTab === 'analysis' && (
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
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm font-semibold text-slate-100">Multi-Agent Analysis</p>
              {(agentStreamEvents.length > 0 || agentStreamResult) && (
                <button
                  onClick={handleExportAgentTranscript}
                  className="shrink-0 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
                >
                  Export transcript
                </button>
              )}
            </div>
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
              {agentStreamResult.result?.specialist_reports && agentStreamResult.result.specialist_reports.length > 0 && (
                <div className="mt-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Specialist breakdown
                  </p>
                  <div className="mb-3 flex flex-wrap gap-2 border-b border-slate-800 pb-3">
                    {agentStreamResult.result.specialist_reports.map((report) => {
                      const domain = report.domain
                      const defaultDomain = agentStreamResult.result.specialist_reports![0].domain
                      const isActive = (activeSpecialistTab ?? defaultDomain) === domain
                      return (
                        <button
                          key={domain}
                          type="button"
                          onClick={() => setActiveSpecialistTab(domain)}
                          className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                            isActive
                              ? 'bg-cyan-600 text-white'
                              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                          }`}
                        >
                          {domain}
                        </button>
                      )
                    })}
                  </div>
                  {(() => {
                    const reports = agentStreamResult.result.specialist_reports!
                    const defaultDomain = reports[0].domain
                    const active = reports.find((r) => r.domain === (activeSpecialistTab ?? defaultDomain)) ?? reports[0]
                    if (!active) return null
                    return active.error ? (
                      <p className="text-sm text-amber-300">{active.error}</p>
                    ) : (
                      <div className="space-y-2">
                        {active.summary && <p className="text-sm text-slate-200">{active.summary}</p>}
                        {active.key_metrics && active.key_metrics.length > 0 && (
                          <ul className="list-inside list-disc text-sm text-slate-300">
                            {active.key_metrics.map((m, i) => (
                              <li key={i}>{m}</li>
                            ))}
                          </ul>
                        )}
                        {active.recommendation && (
                          <p className="text-sm text-indigo-300">{active.recommendation}</p>
                        )}
                      </div>
                    )
                  })()}
                </div>
              )}
              {agentStreamResult.result?.data_summary && (
                <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40">
                  <button
                    type="button"
                    onClick={() => setShowFullDataSummary((v) => !v)}
                    className="flex w-full items-center justify-between px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-200"
                  >
                    <span>Full dataset statistics (exact, computed - not the AI's summary)</span>
                    <span>{showFullDataSummary ? '−' : '+'}</span>
                  </button>
                  {showFullDataSummary && (
                    <pre className="max-h-80 overflow-y-auto whitespace-pre-wrap border-t border-slate-800 px-4 py-3 text-xs leading-relaxed text-slate-300">
                      {agentStreamResult.result.data_summary}
                    </pre>
                  )}
                </div>
              )}
              <p className="mt-3 text-xs text-slate-500">
                Status: {agentStreamResult.status}
                {agentStreamResult.result?.participating_agents && agentStreamResult.result.participating_agents.length > 0 &&
                  ` · Agents: ${agentStreamResult.result.participating_agents.join(', ')}`}
              </p>
            </div>
          )}
        </div>
      )}

      {/* ---------- Chat tab ---------- */}
      {result && activeTab === 'chat' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4 flex items-start justify-between gap-3">
            <div>
              <h3 className="font-medium text-white">Ask your data</h3>
              <p className="mt-1 text-sm text-slate-400">
                {result.kind === 'document'
                  ? 'Ask questions in plain English. Answers are grounded only in retrieved excerpts from this document - never outside knowledge.'
                  : 'Ask questions in plain English. Answers use the exact, full-dataset statistics - never guesses from a sample.'}
              </p>
            </div>
            {messages.length > 0 && (
              <button
                onClick={handleExportChatTranscript}
                className="shrink-0 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
              >
                Export transcript
              </button>
            )}
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 flex flex-col h-[420px]">
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.length === 0 && (
                <p className="text-sm text-slate-500">
                  {result.kind === 'document'
                    ? 'Try asking something like "What does this document say about [a topic]?"'
                    : 'Try asking something like "What\'s the average value in [a numeric column]?" or "Which category appears most often?"'}
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
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-slate-700 pt-2">
                      <span className="text-xs font-medium text-slate-400">Sources:</span>
                      {msg.sources.map((s) => (
                        <button
                          key={s.excerpt_number}
                          type="button"
                          onClick={() => setActiveSource(s)}
                          title={s.preview}
                          className="rounded-full border border-cyan-700/60 bg-cyan-900/30 px-2 py-0.5 text-xs font-medium text-cyan-300 transition hover:border-cyan-500 hover:bg-cyan-800/40 hover:text-cyan-200"
                        >
                          [Excerpt {s.excerpt_number}]
                        </button>
                      ))}
                    </div>
                  )}
                  {msg.role === 'assistant' && msg.consensus && (
                    <div className="mt-2 border-t border-slate-700 pt-2 text-xs text-slate-400">
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span>Confidence: {Math.round(msg.consensus.confidence * 100)}%</span>
                        {msg.consensus.models_used.length > 0 && (
                          <span>Models: {msg.consensus.models_used.join(', ')}</span>
                        )}
                        {msg.consensus.contradictions.length > 0 && (
                          <span className="text-amber-400">
                            {msg.consensus.contradictions.length} contradiction
                            {msg.consensus.contradictions.length === 1 ? '' : 's'} found
                          </span>
                        )}
                        {msg.consensus.synthesis_notes.length > 0 && (
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedReasoning((prev) => {
                                const next = new Set(prev)
                                next.has(i) ? next.delete(i) : next.add(i)
                                return next
                              })
                            }
                            className="font-medium text-cyan-400 hover:text-cyan-300"
                          >
                            {expandedReasoning.has(i) ? 'Hide reasoning' : 'Show reasoning'}
                          </button>
                        )}
                      </div>
                      {expandedReasoning.has(i) && msg.consensus.synthesis_notes.length > 0 && (
                        <ul className="mt-1.5 list-inside list-disc space-y-0.5 text-slate-500">
                          {msg.consensus.synthesis_notes.map((note, n) => (
                            <li key={n}>{note}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
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
      {result && result.kind !== 'document' && activeTab === 'forecast' && (
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

      {/* ---------- AutoML tab (Phase 5) ---------- */}
      {result && result.kind !== 'document' && activeTab === 'automl' && (
        <div className="space-y-4 py-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <h3 className="font-medium text-white">AutoML - train and compare real models</h3>
            <p className="mt-1 text-sm text-slate-400">
              Trains multiple real models (Logistic/Linear Regression, Random Forest, XGBoost, LightGBM),
              cross-validates each, and picks the best by held-out test performance - with real SHAP
              feature importance, not an AI guess.
            </p>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setAutomlMode('train')}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                automlMode === 'train' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Supervised (train &amp; compare)
            </button>
            <button
              onClick={() => setAutomlMode('cluster')}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                automlMode === 'cluster' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Clustering (find groups)
            </button>
            <button
              onClick={() => setAutomlMode('anomaly')}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                automlMode === 'anomaly' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Anomaly detection
            </button>
          </div>

          {automlMode === 'train' && (
          <>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:flex-wrap">
              <label className="text-sm text-slate-400">Target column:</label>
              <select
                value={automlTargetColumn}
                onChange={(e) => setAutomlTargetColumn(e.target.value)}
                className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-blue-500"
              >
                <option value="">Select a column...</option>
                {(result.columns ?? []).map((col) => (
                  <option key={col.name} value={col.name}>
                    {col.name} ({col.dtype})
                  </option>
                ))}
              </select>

              <label className="text-sm text-slate-400">Problem type:</label>
              <select
                value={automlProblemType}
                onChange={(e) => setAutomlProblemType(e.target.value as typeof automlProblemType)}
                className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-blue-500"
              >
                <option value="auto">Auto-detect</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>

              <button
                onClick={handleRunAutoML}
                disabled={automlRunning || !automlTargetColumn}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
              >
                {automlRunning ? 'Training models...' : 'Run AutoML'}
              </button>
            </div>
          </div>

          {automlError && <ErrorBanner message={automlError} />}

          {automlResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Problem type" value={automlResult.problem_type} isText />
                <Stat label="Best model" value={automlResult.best_model_name} isText />
                <Stat label="Rows used" value={automlResult.n_rows_used} />

                <Stat label="Rows dropped" value={automlResult.n_rows_dropped} />
              </div>

              {(automlResult.warnings.length > 0 || automlResult.class_imbalance) && (
                <div className="rounded-xl border border-amber-800 bg-amber-950/30 p-4 space-y-1.5">
                  <h4 className="text-sm font-semibold text-amber-300">Data quality notes</h4>
                  <ul className="list-inside list-disc text-sm text-amber-200/90">
                    {automlResult.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="rounded-xl border border-slate-800 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="text-left p-2">Model</th>
                      <th className="text-left p-2">
                        CV {automlResult.primary_metric} (mean ± std)
                      </th>
                      <th className="text-left p-2">Test metrics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {automlResult.models.map((m) => (
                      <tr
                        key={m.name}
                        className={`border-t border-slate-800 ${
                          m.name === automlResult.best_model_name ? 'bg-blue-950/30' : ''
                        }`}
                      >
                        <td className="p-2 font-medium text-slate-200">
                          {m.name}
                          {m.name === automlResult.best_model_name && (
                            <span className="ml-2 rounded-full bg-blue-600 px-2 py-0.5 text-xs text-white">
                              Best
                            </span>
                          )}
                        </td>
                        <td className="p-2 text-slate-300">
                          {Number.isFinite(m.cv_score_mean) ? `${m.cv_score_mean} ± ${m.cv_score_std}` : '—'}
                        </td>
                        <td className="p-2 text-slate-400">
                          {'error' in m.test_metrics
                            ? String(m.test_metrics.error)
                            : Object.entries(m.test_metrics)
                                .map(([k, v]) => `${k}: ${v}`)
                                .join(' · ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
                <h4 className="mb-2 text-sm font-medium text-slate-300">
                  Top features (SHAP importance)
                </h4>
                {automlResult.shap_importances.length > 0 ? (
                  <div className="space-y-1.5">
                    {automlResult.shap_importances.map((f) => {
                      const maxImportance = automlResult.shap_importances[0].importance || 1
                      const widthPct = Math.max(4, Math.round((f.importance / maxImportance) * 100))
                      return (
                        <div key={f.feature} className="flex items-center gap-2 text-xs">
                          <span className="w-40 truncate text-slate-400">{f.feature}</span>
                          <div className="h-2 flex-1 rounded-full bg-slate-800">
                            <div
                              className="h-2 rounded-full bg-cyan-500"
                              style={{ width: `${widthPct}%` }}
                            />
                          </div>
                          <span className="w-16 text-right text-slate-500">{f.importance}</span>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    {automlResult.shap_unavailable_reason || 'No feature importance available.'}
                  </p>
                )}
              </div>

              <div className="rounded-xl border border-blue-800 bg-blue-950/30 p-4 space-y-2">
                <h4 className="text-sm font-semibold text-blue-300">Business summary</h4>
                <p className="text-sm text-slate-200">{automlResult.business_summary.summary}</p>
                {automlResult.business_summary.key_metrics.length > 0 && (
                  <ul className="list-inside list-disc text-sm text-slate-300">
                    {automlResult.business_summary.key_metrics.map((k, i) => (
                      <li key={i}>{k}</li>
                    ))}
                  </ul>
                )}
                {automlResult.business_summary.recommendation && (
                  <p className="text-sm text-indigo-300">{automlResult.business_summary.recommendation}</p>
                )}
              </div>

              {automlResult.model_id && (
                <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
                  <h4 className="text-sm font-medium text-slate-300">
                    Predict on a new row with the best model ({automlResult.best_model_name})
                  </h4>
                  <p className="text-xs text-slate-500">
                    Leave a field blank to let the model fill it in the same way it handled missing
                    values during training.
                  </p>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {automlResult.feature_columns.map((col) => (
                      <div key={col} className="flex flex-col gap-1">
                        <label className="text-xs text-slate-400">{col}</label>
                        <input
                          type="text"
                          value={automlPredictInputs[col] ?? ''}
                          onChange={(e) =>
                            setAutomlPredictInputs((prev) => ({ ...prev, [col]: e.target.value }))
                          }
                          className="rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 outline-none focus:border-blue-500"
                        />
                      </div>
                    ))}
                  </div>
                  <label className="flex items-center gap-2 text-xs text-slate-400">
                    <input
                      type="checkbox"
                      checked={automlExplainPrediction}
                      onChange={(e) => setAutomlExplainPrediction(e.target.checked)}
                      className="rounded border-slate-600"
                    />
                    Explain this prediction (which features pushed it up or down)
                  </label>
                  <button
                    onClick={handleAutoMLPredict}
                    disabled={automlPredicting}
                    className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
                  >
                    {automlPredicting ? 'Predicting...' : 'Predict'}
                  </button>

                  {automlPredictError && <ErrorBanner message={automlPredictError} />}

                  {automlPredictResult && (
                    <div className="rounded-lg border border-emerald-800 bg-emerald-950/30 p-3 text-sm">
                      <p className="text-emerald-300">
                        Prediction: <span className="font-semibold">{String(automlPredictResult.predictions[0])}</span>
                      </p>
                      {automlPredictResult.probabilities && (
                        <ul className="mt-1 text-xs text-emerald-200/80">
                          {Object.entries(automlPredictResult.probabilities[0]).map(([cls, prob]) => (
                            <li key={cls}>
                              {cls}: {(prob * 100).toFixed(1)}%
                            </li>
                          ))}
                        </ul>
                      )}
                      {automlPredictResult.explanations && automlPredictResult.explanations[0] && (
                        <div className="mt-2 border-t border-emerald-800/50 pt-2">
                          {automlPredictResult.explanations[0].unavailable_reason ? (
                            <p className="text-xs text-slate-500">
                              {automlPredictResult.explanations[0].unavailable_reason}
                            </p>
                          ) : (
                            <>
                              <p className="mb-1 text-xs font-medium text-emerald-200">
                                Why this prediction:
                              </p>
                              <div className="space-y-1">
                                {automlPredictResult.explanations[0].explanation.map((f) => (
                                  <div key={f.feature} className="flex items-center gap-2 text-xs">
                                    <span className="w-32 truncate text-slate-400">{f.feature}</span>
                                    <span className={f.impact >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                      {f.impact >= 0 ? '▲' : '▼'} {Math.abs(f.impact).toFixed(4)}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {automlResult.model_id && (
                <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
                  <h4 className="text-sm font-medium text-slate-300">Batch predict from a CSV</h4>
                  <p className="text-xs text-slate-500">
                    Upload a CSV of new rows - get the same file back with a `prediction` column
                    (and probabilities, for classification) appended.
                  </p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setBatchPredictFile(e.target.files?.[0] ?? null)}
                    className="block w-full text-xs text-slate-400 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-800 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-slate-200 hover:file:bg-slate-700"
                  />
                  <button
                    onClick={handleBatchPredict}
                    disabled={batchPredicting || !batchPredictFile}
                    className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
                  >
                    {batchPredicting ? 'Predicting...' : 'Predict & download CSV'}
                  </button>
                  {batchPredictError && <ErrorBanner message={batchPredictError} />}
                </div>
              )}
            </div>
          )}
          </>
          )}

          {automlMode === 'cluster' && (
          <>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <p className="text-sm text-slate-400">
              Finds natural groupings in the data with KMeans - the number of clusters is chosen
              automatically (silhouette score). No target column needed.
            </p>
            <div className="flex flex-col gap-2">
              <label className="text-sm text-slate-400">
                Feature columns (leave all unchecked to use every numeric column):
              </label>
              <div className="flex flex-wrap gap-2">
                {(result.columns ?? [])
                  .filter((col) => col.dtype.toLowerCase().includes('int') || col.dtype.toLowerCase().includes('float'))
                  .map((col) => {
                    const checked = clusterFeatureColumns.includes(col.name)
                    return (
                      <button
                        key={col.name}
                        type="button"
                        onClick={() =>
                          setClusterFeatureColumns((prev) =>
                            checked ? prev.filter((c) => c !== col.name) : [...prev, col.name]
                          )
                        }
                        className={`rounded-full px-3 py-1 text-xs font-medium ${
                          checked ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        {col.name}
                      </button>
                    )
                  })}
              </div>
            </div>
            <button
              onClick={handleRunClustering}
              disabled={clustering}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
            >
              {clustering ? 'Clustering...' : 'Run clustering'}
            </button>
          </div>

          {clusterError && <ErrorBanner message={clusterError} />}

          {clusterResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Stat label="Clusters found" value={clusterResult.n_clusters} />
                <Stat label="Silhouette score" value={clusterResult.silhouette_score} />
                <Stat label="Features used" value={clusterResult.feature_columns.length} />
              </div>

              <div className="rounded-xl border border-slate-800 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="text-left p-2">Cluster</th>
                      <th className="text-left p-2">Size</th>
                      {clusterResult.feature_columns.map((col) => (
                        <th key={col} className="text-left p-2">
                          {col} (mean)
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(clusterResult.cluster_profiles).map(([clusterId, profile]) => (
                      <tr key={clusterId} className="border-t border-slate-800">
                        <td className="p-2 font-medium text-slate-200">Cluster {clusterId}</td>
                        <td className="p-2 text-slate-300">{clusterResult.cluster_sizes[clusterId]}</td>
                        {clusterResult.feature_columns.map((col) => (
                          <td key={col} className="p-2 text-slate-400">
                            {profile[col]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          </>
          )}

          {automlMode === 'anomaly' && (
          <>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 space-y-3">
            <p className="text-sm text-slate-400">
              Flags unusual/outlier rows with IsolationForest - a sibling to Clustering, but
              finds the rows that DON'T fit any group well instead of grouping similar rows
              together. No target column needed.
            </p>
            <div className="flex flex-col gap-2">
              <label className="text-sm text-slate-400">
                Feature columns (leave all unchecked to use every numeric column):
              </label>
              <div className="flex flex-wrap gap-2">
                {(result.columns ?? [])
                  .filter((col) => col.dtype.toLowerCase().includes('int') || col.dtype.toLowerCase().includes('float'))
                  .map((col) => {
                    const checked = anomalyFeatureColumns.includes(col.name)
                    return (
                      <button
                        key={col.name}
                        type="button"
                        onClick={() =>
                          setAnomalyFeatureColumns((prev) =>
                            checked ? prev.filter((c) => c !== col.name) : [...prev, col.name]
                          )
                        }
                        className={`rounded-full px-3 py-1 text-xs font-medium ${
                          checked ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        {col.name}
                      </button>
                    )
                  })}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm text-slate-400">
                Expected anomaly rate: {Math.round(anomalyContamination * 100)}%
              </label>
              <input
                type="range"
                min={1}
                max={30}
                value={Math.round(anomalyContamination * 100)}
                onChange={(e) => setAnomalyContamination(Number(e.target.value) / 100)}
                className="w-40"
              />
            </div>
            <button
              onClick={handleDetectAnomalies}
              disabled={detectingAnomalies}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
            >
              {detectingAnomalies ? 'Detecting...' : 'Detect anomalies'}
            </button>
          </div>

          {anomalyError && <ErrorBanner message={anomalyError} />}

          {anomalyResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Stat label="Rows analyzed" value={anomalyResult.n_rows_analyzed} />
                <Stat label="Anomalies found" value={anomalyResult.n_anomalies} />
                <Stat label="Anomaly rate" value={`${Math.round(anomalyResult.anomaly_rate * 100)}%`} isText />
              </div>

              {anomalyResult.anomalies.length > 0 && (
                <div className="rounded-xl border border-slate-800 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-900 text-slate-400">
                      <tr>
                        <th className="text-left p-2">Row</th>
                        <th className="text-left p-2">Anomaly score</th>
                        {anomalyResult.feature_columns.map((col) => (
                          <th key={col} className="text-left p-2">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {anomalyResult.anomalies.map((a) => (
                        <tr key={a.row_index} className="border-t border-slate-800">
                          <td className="p-2 font-medium text-slate-200">{a.row_index}</td>
                          <td className="p-2 text-amber-400">{a.anomaly_score}</td>
                          {anomalyResult.feature_columns.map((col) => (
                            <td key={col} className="p-2 text-slate-400">
                              {a.values[col]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {anomalyResult.anomalies_truncated && (
                    <p className="p-2 text-xs text-slate-500">
                      Showing the {anomalyResult.anomalies.length} most anomalous rows.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
          </>
          )}

          {automlVersionsError && <ErrorBanner message={automlVersionsError} />}

          {automlVersionsLoading && automlVersions.length === 0 && (
            <p className="text-xs text-slate-500">Loading AutoML history...</p>
          )}

          {automlVersions.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <h4 className="mb-2 text-sm font-medium text-slate-300">
                Run history ({automlVersions.length} run{automlVersions.length === 1 ? '' : 's'})
              </h4>
              <div className="divide-y divide-slate-800">
                {automlVersions.map((v) => (
                  <div key={v.id} className="flex flex-wrap items-center justify-between gap-1 py-2 text-sm">
                    <div>
                      <span className="font-medium text-slate-200">Run {v.version_number}</span>
                      <span className="ml-2 text-xs text-slate-500">
                        {new Date(v.created_at).toLocaleString()} &middot; {v.problem_type} on "{v.target_column}"
                        &middot; best: {v.best_model_name}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Past runs show metrics only - a trained model can only be used for new predictions for
                1 hour after training, so older runs can't be re-predicted on without running again.
              </p>
            </div>
          )}
        </div>
      )}

      {/* ---------- Clean Data tab ---------- */}
      {result && result.kind !== 'document' && activeTab === 'clean' && (
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
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className="text-sm font-medium text-emerald-300">Before vs After</h4>
                <button
                  onClick={() => handleDownloadCleaned()}
                  disabled={cleanDownloading}
                  className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  {cleanDownloading ? 'Downloading...' : 'Download cleaned CSV'}
                </button>
              </div>
              {cleanDownloadError && <ErrorBanner message={cleanDownloadError} />}
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

          {cleaningVersionsError && <ErrorBanner message={cleaningVersionsError} />}

          {cleaningVersionsLoading && cleaningVersions.length === 0 && (
            <p className="text-xs text-slate-500">Loading cleaning history...</p>
          )}

          {cleaningVersions.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <h4 className="mb-2 text-sm font-medium text-slate-300">
                Cleaning history ({cleaningVersions.length} version{cleaningVersions.length === 1 ? '' : 's'})
              </h4>
              <div className="divide-y divide-slate-800">
                {cleaningVersions.map((v) => (
                  <div key={v.id} className="flex items-center justify-between py-2 text-sm">
                    <div>
                      <span className="font-medium text-slate-200">Version {v.version_number}</span>
                      <span className="ml-2 text-xs text-slate-500">
                        {new Date(v.created_at).toLocaleString()} &middot; {v.report.after.row_count} rows,{' '}
                        {v.report.after.missing_values} missing, {v.report.after.duplicate_rows} duplicates
                      </span>
                    </div>
                    <button
                      onClick={() => handleDownloadCleaned(v.id, v.version_number)}
                      disabled={cleanDownloading}
                      className="rounded-lg bg-slate-800 px-3 py-1 text-xs font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-50"
                    >
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---------- EDA & Charts tab ---------- */}
      {result && result.kind !== 'document' && activeTab === 'eda' && (
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
      {result && result.kind !== 'document' && activeTab === 'visualize' && (
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
                    {(result.columns ?? []).map((col) => (
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
                    {(result.columns ?? []).map((col) => (
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

      {/* ---------- Usage / cost savings dashboard ---------- */}
      {activeTab === 'usage' && (
        <div className="space-y-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium text-white">Usage &amp; savings</h3>
              <p className="text-sm text-slate-400">
                App-wide (not per-user), since this app started running. Domain Gate and Evidence Gate
                block off-topic or unsupported questions before any LLM call happens; caching returns
                repeated questions instantly; fast-tier consensus uses one model with fallback instead
                of querying all three providers.
              </p>
            </div>
            <button
              onClick={loadUsageStats}
              disabled={usageLoading}
              className="text-xs text-slate-400 hover:text-slate-200 disabled:opacity-50"
            >
              {usageLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>

          {usageError && <ErrorBanner message={usageError} />}

          {usageStats && (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="LLM calls avoided" value={usageStats.total_calls_avoided} />
                <Stat
                  label="Cache hit rate"
                  value={usageStats.cache_hit_rate !== null ? `${Math.round(usageStats.cache_hit_rate * 100)}%` : 'n/a'}
                  isText
                />
                <Stat label="Fast-tier calls" value={usageStats.counters.consensus_calls_fast} />
                <Stat label="Full-tier calls" value={usageStats.counters.consensus_calls_full} />
              </div>

              <div className="rounded-xl border border-slate-800 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="text-left p-2">Metric</th>
                      <th className="text-left p-2">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-300">Cache hits</td>
                      <td className="p-2 text-slate-400">{usageStats.counters.cache_hits}</td>
                    </tr>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-300">Cache misses</td>
                      <td className="p-2 text-slate-400">{usageStats.counters.cache_misses}</td>
                    </tr>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-300">Domain Gate rejections (no LLM call)</td>
                      <td className="p-2 text-slate-400">{usageStats.counters.domain_gate_rejections}</td>
                    </tr>
                    <tr className="border-t border-slate-800">
                      <td className="p-2 text-slate-300">Evidence Gate rejections (no LLM call)</td>
                      <td className="p-2 text-slate-400">{usageStats.counters.evidence_gate_rejections}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="rounded-xl border border-emerald-800 bg-emerald-950/30 p-4 space-y-1.5">
                <h4 className="text-sm font-semibold text-emerald-300">
                  Estimated tokens saved: {usageStats.estimated_tokens_saved.total.toLocaleString()}
                </h4>
                <p className="text-xs text-emerald-200/70">{usageStats.estimated_tokens_saved.methodology}</p>
              </div>
            </>
          )}
        </div>
      )}

      <SourceDrawer source={activeSource} onClose={() => setActiveSource(null)} />
    </section>
  )
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-800 bg-red-950/30 px-3 py-2 text-sm text-red-300">{message}</div>
  )
}

function downloadTextFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

function agentEventIcon(type: string): string {
  switch (type) {
    case 'domain_gate_start':
      return '🚦'
    case 'domain_gate_passed':
      return '✅'
    case 'domain_gate_rejected':
      return '🚫'
    case 'input_security_start':
      return '🛂'
    case 'input_security_passed':
      return '✅'
    case 'input_security_blocked':
      return '🚫'
    case 'manager_start':
    case 'plan_created':
    case 'plan_ordered':
      return '🧭'
    case 'manager_planning_failed':
    case 'planner_validation_failed':
      return '⚠'
    case 'manager_replanning':
    case 'manager_replan_triggered':
      return '🔄'
    case 'manager_replan_exhausted':
      return '⚠'
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

function SourceDrawer({ source, onClose }: { source: ChatSource | null; onClose: () => void }) {
  useEffect(() => {
    if (!source) return
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [source, onClose])

  if (!source) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} aria-hidden="true" />
      <div className="relative flex h-full w-full max-w-md flex-col border-l border-slate-800 bg-slate-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <h4 className="text-sm font-semibold text-white">Excerpt {source.excerpt_number}</h4>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg px-2 py-1 text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            ✕
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-4">
          {typeof source.score === 'number' && (
            <p className="mb-3 text-xs text-slate-500">Relevance score: {source.score.toFixed(3)}</p>
          )}
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
            {source.text || source.preview}
          </p>
        </div>
      </div>
    </div>
  )
}