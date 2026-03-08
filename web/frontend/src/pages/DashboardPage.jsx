import { useEffect, useState } from 'react'
import { Link, useNavigate }   from 'react-router-dom'
import { useAuth }             from '../context/AuthContext'
import { sessions }            from '../api/client'
import { OVERALL_RATING_COLOURS } from '../utils/constants'
import { primaryBtn }          from '../utils/styles'
import Spinner                 from '../components/shared/Spinner'

export default function DashboardPage() {
  const { user }                    = useAuth()
  const navigate                    = useNavigate()
  const [history, setHistory]       = useState([])
  const [loading, setLoading]       = useState(true)

  useEffect(() => {
    sessions.list(10)
      .then(setHistory)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const completed = history.filter(s => s.exercise_completed).length
  const streak    = computeStreak(history)

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 6 }}>
          Welcome back, {user?.full_name?.split(' ')[0]} 👋
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 15 }}>
          Track your progress and start your next therapy session.
        </p>
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 36 }}>
        <StatCard icon="📋" label="Total Sessions" value={history.length} />
        <StatCard icon="✅" label="Completed"      value={completed}      />
        <StatCard icon="🔥" label="Day Streak"     value={streak}         />
      </div>

      {/* CTA */}
      <div style={{
        background:   'linear-gradient(135deg, rgba(99,179,237,0.12) 0%, rgba(79,209,197,0.07) 100%)',
        border:       '1px solid var(--border-accent)',
        borderRadius: 'var(--radius)',
        padding:      '28px 32px',
        marginBottom: 36,
        display:      'flex',
        alignItems:   'center',
        justifyContent: 'space-between',
        gap:          20,
      }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Ready for today's session?</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
            Each session adapts to your current speech profile for maximum benefit.
          </p>
        </div>
        <button onClick={() => navigate('/session')} style={{ ...primaryBtn, width: 'auto', whiteSpace: 'nowrap' }}>
          Start Session →
        </button>
      </div>

      {/* Session history */}
      <section>
        <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Session History</h2>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}><Spinner size={28} /></div>
        ) : history.length === 0 ? (
          <EmptyState />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {history.map(s => <SessionRow key={s.id} session={s} />)}
          </div>
        )}
      </section>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ icon, label, value }) {
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '20px 24px',
    }}>
      <div style={{ fontSize: 26, marginBottom: 8 }}>{icon}</div>
      <div style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>{value}</div>
      <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</div>
    </div>
  )
}

function SessionRow({ session: s }) {
  const colour = OVERALL_RATING_COLOURS[s.overall_rating] ?? 'var(--text-muted)'
  const date   = new Date(s.created_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })

  return (
    <Link to={`/session/${s.id}`} style={{ textDecoration: 'none' }}>
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12,
        padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 16,
        transition: 'border-color 0.15s', cursor: 'pointer',
      }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-accent)'}
        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
      >
        <div style={{
          width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
          background: s.exercise_completed ? 'rgba(104,211,145,0.15)' : 'var(--bg-elevated)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16,
        }}>
          {s.exercise_completed ? '✅' : '🕐'}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>
            {s.recommended_exercise_name ?? 'Assessment only'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{date}</div>
        </div>
        {s.overall_rating && (
          <span style={{
            fontSize: 11, fontWeight: 600, padding: '3px 10px', borderRadius: 20,
            background: `${colour}22`, color: colour, border: `1px solid ${colour}44`,
            textTransform: 'capitalize', letterSpacing: 0.4,
          }}>
            {s.overall_rating.replace('_', ' ')}
          </span>
        )}
        <span style={{ fontSize: 18, color: 'var(--text-muted)' }}>›</span>
      </div>
    </Link>
  )
}

function EmptyState() {
  const navigate = useNavigate()
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
      padding: '48px 32px', textAlign: 'center',
    }}>
      <div style={{ fontSize: 40, marginBottom: 16 }}>🎙</div>
      <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>No sessions yet</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 24 }}>
        Start your first session to see your progress here.
      </p>
      <button onClick={() => navigate('/session')} style={{ ...primaryBtn, width: 'auto', display: 'inline-block' }}>
        Start First Session →
      </button>
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function computeStreak(sessions) {
  if (!sessions.length) return 0
  const dates = [...new Set(
    sessions.map(s => new Date(s.created_at).toDateString())
  )].map(d => new Date(d)).sort((a, b) => b - a)

  let streak = 1
  for (let i = 1; i < dates.length; i++) {
    const diff = (dates[i - 1] - dates[i]) / 86400000
    if (diff === 1) streak++
    else break
  }
  return streak
}
