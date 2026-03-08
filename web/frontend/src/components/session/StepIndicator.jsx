import { SESSION_STEPS } from '../../utils/constants'

export default function StepIndicator({ step }) {
  const idx = SESSION_STEPS.findIndex(s => s.id === step)
  return (
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 32 }}>
      {SESSION_STEPS.map((s, i) => (
        <div key={s.id} style={{ display: 'flex', alignItems: 'center', flex: i < SESSION_STEPS.length - 1 ? 1 : 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0, opacity: i <= idx ? 1 : 0.3 }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, fontWeight: 700,
              background: i < idx ? 'var(--accent-green)' : i === idx ? 'var(--accent-blue)' : 'var(--bg-elevated)',
              color: i <= idx ? 'white' : 'var(--text-muted)',
              border: i === idx ? '2px solid rgba(99,179,237,0.4)' : '2px solid transparent',
            }}>
              {i < idx ? '✓' : i + 1}
            </div>
            <span style={{ fontSize: 13, fontWeight: i === idx ? 600 : 400, color: i === idx ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
              {s.label}
            </span>
          </div>
          {i < SESSION_STEPS.length - 1 && (
            <div style={{ flex: 1, height: 1, margin: '0 12px', background: i < idx ? 'var(--accent-green)' : 'var(--border)' }} />
          )}
        </div>
      ))}
    </div>
  )
}
