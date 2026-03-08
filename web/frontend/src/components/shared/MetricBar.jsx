/**
 * Horizontal progress bar for a single discourse metric.
 * @param {{ label: string, value: number, color: string }} props
 */
export default function MetricBar({ label, value, color }) {
  const pct = Math.min(Math.max(value * 100, 0), 100)
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontSize: 13, color, fontWeight: 600 }}>{pct.toFixed(0)}%</span>
      </div>
      <div style={{ height: 6, background: 'var(--bg-elevated)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: 3, transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  )
}
