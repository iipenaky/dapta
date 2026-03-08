export default function ErrorBanner({ msg, onClose }) {
  return (
    <div role="alert" style={{
      background: 'rgba(252,129,129,0.1)', border: '1px solid rgba(252,129,129,0.3)',
      borderRadius: 8, padding: '12px 16px', marginBottom: 20,
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <span style={{ fontSize: 13, color: 'var(--accent-red)' }}>⚠️ {msg}</span>
      <button onClick={onClose} aria-label="Dismiss"
        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 16 }}>
        ✕
      </button>
    </div>
  )
}
