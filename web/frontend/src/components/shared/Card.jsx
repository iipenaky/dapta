// Card.jsx
export default function Card({ title, subtitle, children }) {
  return (
    <div style={{
      background:   'var(--bg-card)',
      border:       '1px solid var(--border)',
      borderRadius: 'var(--radius)',
      padding:      '28px 32px',
      boxShadow:    'var(--shadow-card)',
      animation:    'fadeIn 0.3s ease',
    }}>
      {title && (
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 22, marginBottom: subtitle ? 6 : 24 }}>
          {title}
        </h3>
      )}
      {subtitle && (
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 24 }}>
          {subtitle}
        </p>
      )}
      {children}
    </div>
  )
}
