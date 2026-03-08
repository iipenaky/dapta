/**
 * Shared shell for Login and Signup pages.
 * Provides the centred card layout, logo, title, and error display.
 */
export default function AuthForm({ title, subtitle, error, onSubmit, children }) {
  return (
    <div style={{
      minHeight:      '100vh',
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'center',
      padding:        '24px 16px',
      background:     'var(--bg-root)',
    }}>
      <div style={{ width: '100%', maxWidth: 420 }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{ fontSize: 40, marginBottom: 10 }}>🧠</div>
          <h1 style={{ fontSize: 26, fontWeight: 800, color: 'var(--text-primary)' }}>DAPTA</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            Discourse-Aware Personalised Therapy
          </p>
        </div>

        {/* Card */}
        <div style={{
          background:   'var(--bg-card)',
          border:       '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding:      '32px',
          boxShadow:    'var(--shadow-card)',
        }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>{title}</h2>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 24 }}>{subtitle}</p>

          {error && (
            <div style={{
              background: 'rgba(252,129,129,0.1)', border: '1px solid rgba(252,129,129,0.3)',
              borderRadius: 8, padding: '10px 14px', marginBottom: 20,
              fontSize: 13, color: 'var(--accent-red)',
            }}>
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={onSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {children}
          </form>
        </div>
      </div>
    </div>
  )
}
