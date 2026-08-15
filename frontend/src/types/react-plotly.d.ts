declare module 'react-plotly.js' {
  import * as React from 'react'

  interface PlotParams {
    data: unknown[]
    layout?: Record<string, unknown>
    config?: Record<string, unknown>
    frames?: unknown[]
    style?: React.CSSProperties
    className?: string
    useResizeHandler?: boolean
    divId?: string
    onInitialized?: (figure: unknown, graphDiv: HTMLElement) => void
    onUpdate?: (figure: unknown, graphDiv: HTMLElement) => void
    onError?: (err: unknown) => void
  }

  export default class Plot extends React.Component<PlotParams> {}
}

// Lighter-weight factory pattern (react-plotly.js/factory + plotly.js-dist-min)
// avoids a "process is not defined" runtime error that the full 'plotly.js'
// package can trigger under Vite, since dist-min ships a pre-bundled browser build.
declare module 'react-plotly.js/factory' {
  export default function createPlotlyComponent(plotly: unknown): unknown
}

declare module 'plotly.js-dist-min' {
  const Plotly: {
    downloadImage: (graphDiv: unknown, opts: Record<string, unknown>) => Promise<string>
    [key: string]: unknown
  }
  export default Plotly
}