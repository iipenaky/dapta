import { Card, MetricBar, SkeletonLoader } from '../shared'
import { METRIC_LIST }                     from '../../utils/constants'
import { primaryBtn }                      from '../../utils/styles'

export default function RecommendStep({ result, loading, onGetRec }) {
  const metrics = result?.metrics ?? {}
  return (
    <Card title="Your Speech Metrics" subtitle="Here is what DAPTA found in your speech today.">
      {result?.transcript && (
        <div style={{
          background: 'var(--bg-primary)', borderRadius: 8, padding: '12px 16px',
          fontSize: 13, color: 'var(--text-secondary)', marginBottom: 24,
          border: '1px solid var(--border)', lineHeight: 1.6, fontStyle: 'italic',
        }}>
          "{result.transcript.length > 200 ? `${result.transcript.slice(0, 200)}…` : result.transcript}"
        </div>
      )}

      {loading ? (
        <SkeletonLoader rows={5} message="Finding your personalised exercise…" />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 28 }}>
          {METRIC_LIST.map(m => (
            <MetricBar key={m.key} label={m.label} value={metrics[m.key] ?? 0} color={m.color} />
          ))}
        </div>
      )}

      <div style={{
        background: 'rgba(99,179,237,0.06)', borderRadius: 10, padding: '16px 18px', marginBottom: 24,
      }}>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
          DAPTA will now use your speech profile and session history to select the most
          beneficial exercise for your recovery stage.
        </p>
      </div>

      <button onClick={onGetRec} disabled={loading} style={primaryBtn}>
        {loading ? 'Finding your exercise…' : 'Get My Exercise Recommendation →'}
      </button>
    </Card>
  )
}
