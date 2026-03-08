export default function SkeletonLoader({ rows = 4, message = 'Analysing your speech…' }) {
  return (
    <div style={{ padding: '8px 0' }} aria-busy="true" aria-label={message}>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 20, animation: 'pulse 1.5s ease-in-out infinite' }}>
        {message}
      </p>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{
          height: 14, borderRadius: 6, background: 'var(--bg-elevated)', marginBottom: 12,
          width: `${85 - i * 8}%`, animation: `pulse 1.5s ease-in-out ${i * 0.1}s infinite`,
        }} />
      ))}
    </div>
  )
}
