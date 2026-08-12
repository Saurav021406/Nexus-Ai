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
