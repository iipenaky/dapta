export default function TabBar({ tabs, active, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 4, background: 'var(--bg-primary)', borderRadius: 10, padding: 4 }}>
      {tabs.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)} style={{
          flex: 1, padding: '9px 12px', border: 'none', borderRadius: 8, cursor: 'pointer',
          fontFamily: 'var(--font-body)', fontSize: 13, fontWeight: 500, transition: 'all 0.2s',
          background: active === t.id ? 'var(--bg-elevated)' : 'transparent',
          color:      active === t.id ? 'var(--text-primary)' : 'var(--text-secondary)',
        }}>
          {t.label}
        </button>
      ))}
    </div>
  )
}
