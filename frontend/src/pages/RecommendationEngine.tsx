import { useState } from 'react'
import { apiFetch } from '../lib/api'

interface RecommendationColumn {
  name: string
  dtype: string
}

interface Recommendation {
  item: string
  score: number
  similar_customer_count: number
}

interface RecommendationResult {
  customer_value: string
  item_column: string
  method: string
  interaction_count: number
  recommendations: Recommendation[]
}

export default function RecommendationEngine({
  datasetId,
  columns,
}: {
  datasetId: string
  columns: RecommendationColumn[]
}) {
  const [userColumn, setUserColumn] = useState('')
  const [userValue, setUserValue] = useState('')
  const [itemColumn, setItemColumn] = useState('')
  const [ratingColumn, setRatingColumn] = useState('')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<RecommendationResult | null>(null)

  async function generateRecommendations(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const response = await apiFetch('/recommendations/generate', {
        method: 'POST',
        body: JSON.stringify({
          dataset_id: datasetId,
          user_column: userColumn,
          user_value: userValue.trim(),
          item_column: itemColumn,
          rating_column: ratingColumn || null,
        }),
      })
      setResult(response as RecommendationResult)
    } catch (requestError) {
      setError((requestError as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="nexus-tool">
      <div className="nexus-tool__intro">
        <h3 className="text-lg font-semibold text-white">Recommendation Engine</h3>
        <p className="mt-1 text-sm text-slate-400">
          Find products a customer may like from customer-product interactions in this dataset. Recommendations
          are ranked using products interacted with by similar customers.
        </p>
      </div>

      <form onSubmit={generateRecommendations} className="nexus-tool__form space-y-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-300">
            Customer column
            <select
              required
              value={userColumn}
              onChange={(event) => setUserColumn(event.target.value)}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
            >
              <option value="">Select a customer identifier...</option>
              {columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
            </select>
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Customer value
            <input
              required
              value={userValue}
              onChange={(event) => setUserValue(event.target.value)}
              placeholder="Enter the exact value from that column"
              maxLength={500}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder:text-slate-600"
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Product / item column
            <select
              required
              value={itemColumn}
              onChange={(event) => setItemColumn(event.target.value)}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
            >
              <option value="">Select a product column...</option>
              {columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
            </select>
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Rating column <span className="text-slate-500">(optional)</span>
            <select
              value={ratingColumn}
              onChange={(event) => setRatingColumn(event.target.value)}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
            >
              <option value="">No rating column (implicit interactions)</option>
              {columns.filter((column) => column.dtype.toLowerCase() === 'numeric')
                .map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
            </select>
          </label>
        </div>
        <p className="text-xs text-slate-500">
          Use the exact customer value as it appears in the uploaded data. If ratings are provided, higher
          ratings carry more weight.
        </p>
        <button
          type="submit"
          disabled={running || !userColumn || !itemColumn || !userValue.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? 'Finding recommendations...' : 'Find recommendations'}
        </button>
      </form>

      {error && <p role="alert" className="rounded-lg border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">{error}</p>}

      {result && (
        <div className="nexus-surface space-y-3 p-5">
          <div>
            <h4 className="font-medium text-white">Recommended {result.item_column} values</h4>
            <p className="mt-1 text-xs text-slate-500">
              Based on {result.interaction_count} interaction(s) for customer {result.customer_value}.
            </p>
          </div>
          {result.recommendations.map((recommendation, index) => (
            <div key={recommendation.item} className="flex items-center justify-between gap-4 rounded-xl border border-white/[0.07] bg-white/[0.02] p-3">
              <div className="flex min-w-0 items-center gap-3">
                <span className="text-sm font-semibold text-cyan-300">{index + 1}.</span>
                <span className="truncate text-sm text-slate-200">{recommendation.item}</span>
              </div>
              <span className="shrink-0 text-xs text-slate-500">
                {recommendation.similar_customer_count} similar customer(s)
              </span>
            </div>
          ))}
          <p className="text-xs text-slate-500">{result.method}</p>
        </div>
      )}
    </div>
  )
}
