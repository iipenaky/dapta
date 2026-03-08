import { Card }                              from '../shared'
import { METRIC_LIST, OVERALL_RATING_COLOURS } from '../../utils/constants'
import { primaryBtn, secondaryBtn, sectionTitle } from '../../utils/styles'

export default function FeedbackStep({ feedback, onNewSession, onDashboard }) {
  const f      = feedback?.feedback ?? {}
  const colour = OVERALL_RATING_COLOURS[f.overall] ?? 'var(--accent-blue)'

  return (
    <Card title="Session Complete" subtitle="Here is how you did today.">
      {/* Hero */}
      <div style={{ textAlign: 'center', padding: '28px 0', borderBottom: '1px solid var(--border)', marginBottom: 24 }}>
        <div style={{ fontSize: 56, marginBottom: 12 }}>{f.emoji ?? '✅'}</div>
        <p style={{ fontSize: 16, color: colour, fontWeight: 600, marginBottom: 8 }}>{f.message}</p>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{f.encouragement}</p>
      </div>

      {/* Metric comparison */}
      {feedback?.metrics_before && feedback?.metrics_after && (
        <section style={{ marginBottom: 24 }}>
          <h4 style={sectionTitle}>Metric Changes</h4>
          {METRIC_LIST.map(m => {
            const b = feedback.metrics_before[m.key] ?? 0
            const a = feedback.metrics_after[m.key]  ?? 0
            const d = a - b
            const c = d > 0.05 ? 'var(--accent-green)' : d < -0.05 ? 'var(--accent-red)' : 'var(--text-muted)'
            return (
              <div key={m.key} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)', width: 140, flexShrink: 0 }}>{m.label}</span>
                <div style={{ flex: 1, height: 6, background: 'var(--bg-primary)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(a * 100, 100)}%`, height: '100%', background: c, borderRadius: 3, transition: 'width 0.6s ease' }} />
                </div>
                <span style={{ fontSize: 12, color: c, width: 50, textAlign: 'right' }}>
                  {d > 0 ? '+' : ''}{(d * 100).toFixed(0)}%
                </span>
              </div>
            )
          })}
        </section>
      )}

      {/* Transcript */}
      {feedback?.transcript && (
        <section style={{ marginBottom: 24 }}>
          <h4 style={sectionTitle}>What You Said</h4>
          <div style={{
            background: 'var(--bg-primary)', borderRadius: 8, padding: '12px 16px',
            fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7,
            border: '1px solid var(--border)', fontStyle: 'italic',
          }}>
            "{feedback.transcript}"
          </div>
        </section>
      )}

      {/* Improved list */}
      {f.improved?.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <p style={{ fontSize: 13, color: 'var(--accent-green)', fontWeight: 500, marginBottom: 4 }}>✓ Improved today</p>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{f.improved.join(' · ')}</p>
        </div>
      )}

      <div style={{ display: 'flex', gap: 12, marginTop: 28 }}>
        <button onClick={onNewSession} style={{ ...primaryBtn, flex: 1 }}>Start New Session</button>
        <button onClick={onDashboard}  style={secondaryBtn}>Back to Dashboard</button>
      </div>
    </Card>
  )
}
