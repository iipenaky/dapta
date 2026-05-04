import { useEffect, useState }  from 'react'
import { Link, useNavigate }     from 'react-router-dom'
import { useAuth }               from '../context/AuthContext'
import { sessions }              from '../api/client'
import { OVERALL_RATING_COLOURS } from '../utils/constants'
import { primaryBtn }            from '../utils/styles'
import Spinner                   from '../components/shared/Spinner'

/*  Google Fonts injection  */
const fontLink = document.createElement('link')
fontLink.rel  = 'stylesheet'
fontLink.href = 'https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&display=swap'
document.head.appendChild(fontLink)

/*  Design tokens  */
const T = {
  cream:   '#F5F0E8',
  paper:   '#EDE8DC',
  forest:  '#1E3A2F',
  sage:    '#4A7B5F',
  moss:    '#6B9E7E',
  sand:    '#C4A882',
  amber:   '#D4843A',
  ink:     '#1A1A1A',
  muted:   '#7A7268',
  border:  'rgba(30,58,47,0.12)',
  cardBg:  'rgba(255,255,255,0.72)',
  radius:  '14px',
}

const css = `
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes countUp {
    from { opacity:0; transform:translateY(8px) scale(0.9); }
    to   { opacity:1; transform:translateY(0) scale(1); }
  }
  @keyframes waveIn {
    from { transform:scaleX(0); }
    to   { transform:scaleX(1); }
  }
  .dash-root * { font-family:'DM Sans', sans-serif; box-sizing:border-box; }
  .stat-card { transition: transform 0.2s ease, box-shadow 0.2s ease; }
  .stat-card:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(30,58,47,0.14); }
  .session-row { transition: all 0.18s ease; }
  .session-row:hover { transform:translateX(4px); background:rgba(255,255,255,0.95) !important; box-shadow:0 4px 24px rgba(30,58,47,0.1); }
  .start-btn {
    background: ${T.forest};
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 13px 28px;
    font-size: 14px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    letter-spacing: 0.3px;
    transition: background 0.2s ease, transform 0.15s ease;
  }
  .start-btn:hover { background:${T.sage}; transform:translateY(-1px); }
  .badge {
    display:inline-flex; align-items:center;
    font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px;
    text-transform:capitalize; letter-spacing:0.5px;
  }
`

export default function DashboardPage() {
  const { user }              = useAuth()
  const navigate              = useNavigate()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    sessions.list(10)
      .then(setHistory)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60)
    return () => clearTimeout(t)
  }, [])

  const completed = history.filter(s => s.exercise_completed).length
  const streak    = computeStreak(history)
  const firstName = user?.full_name?.split(' ')[0] ?? 'there'

  return (
    <div className="dash-root" style={{
      minHeight: '100vh',
      background: `${T.cream}`,
      backgroundImage: `
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(74,123,95,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 90%, rgba(212,132,58,0.06) 0%, transparent 55%)
      `,
    }}>
      <style>{css}</style>

      <div style={{ maxWidth: 860, margin: '0 auto', padding: '48px 24px 80px' }}>

        {/*  Header  */}
        <header style={{
          marginBottom: 48,
          animation: visible ? 'fadeUp 0.55s ease both' : 'none',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:10 }}>
            <span style={{
              display:'inline-block', width:28, height:3,
              background: T.sage, borderRadius:2,
              transformOrigin:'left',
              animation: visible ? 'waveIn 0.6s 0.2s ease both' : 'none',
            }} />
            <span style={{ fontSize:12, fontWeight:600, color:T.sage, letterSpacing:1.8, textTransform:'uppercase' }}>
              Dashboard
            </span>
          </div>
          <h1 style={{
            fontFamily: 'Lora, serif',
            fontSize: 36,
            fontWeight: 700,
            color: T.ink,
            margin: 0,
            lineHeight: 1.2,
            letterSpacing: '-0.5px',
          }}>
            Good to see you, <em style={{ fontStyle:'italic', color:T.sage }}>{firstName}</em>
          </h1>
          <p style={{ marginTop:8, color:T.muted, fontSize:15, fontWeight:400 }}>
            Your personalised speech therapy — pick up where you left off.
          </p>
        </header>

        {/*  Stats  */}
        <div style={{
          display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:14, marginBottom:36,
        }}>
          {[
            { icon:'📋', label:'Total Sessions', value:history.length, delay:'0.1s' },
            { icon:'✅', label:'Completed',      value:completed,       delay:'0.2s' },
            { icon:'🔥', label:'Day Streak',     value:streak,          delay:'0.3s' },
          ].map(({ icon, label, value, delay }) => (
            <StatCard key={label} icon={icon} label={label} value={value} visible={visible} delay={delay} />
          ))}
        </div>

        {/*  CTA Banner  */}
        <div style={{
          position: 'relative',
          overflow: 'hidden',
          background: T.forest,
          borderRadius: 18,
          padding: '32px 36px',
          marginBottom: 44,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 20,
          animation: visible ? 'fadeUp 0.55s 0.35s ease both' : 'none',
          opacity: visible ? undefined : 0,
        }}>
          {/* Decorative circles */}
          <div style={{
            position:'absolute', right:-40, top:-40,
            width:180, height:180, borderRadius:'50%',
            border:`1px solid rgba(255,255,255,0.07)`,
            pointerEvents:'none',
          }} />
          <div style={{
            position:'absolute', right:20, top:-80,
            width:240, height:240, borderRadius:'50%',
            border:`1px solid rgba(255,255,255,0.04)`,
            pointerEvents:'none',
          }} />

          <div style={{ position:'relative' }}>
            <div style={{
              fontSize:11, fontWeight:600, color:T.moss,
              letterSpacing:1.6, textTransform:'uppercase', marginBottom:8,
            }}>
              Today's Session
            </div>
            <h2 style={{
              fontFamily:'Lora, serif',
              fontSize:22, fontWeight:600,
              color:'#fff', margin:'0 0 6px',
              letterSpacing:'-0.3px',
            }}>
              Ready to practise?
            </h2>
            <p style={{ color:'rgba(255,255,255,0.5)', fontSize:13, margin:0, maxWidth:320 }}>
              Each session adapts in real time to your current speech profile.
            </p>
          </div>

          <button className="start-btn" onClick={() => navigate('/session')} style={{
            background: T.sand,
            color: T.forest,
            flexShrink:0,
            fontSize:13,
          }}>
            Start Session →
          </button>
        </div>

        {/*  History  */}
        <section style={{ animation: visible ? 'fadeUp 0.55s 0.45s ease both' : 'none', opacity: visible ? undefined : 0 }}>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:18 }}>
            <h2 style={{
              fontFamily:'Lora, serif',
              fontSize:19, fontWeight:600,
              color:T.ink, margin:0, letterSpacing:'-0.2px',
            }}>
              Session History
            </h2>
            {history.length > 0 && (
              <span style={{ fontSize:12, color:T.muted }}>{history.length} sessions</span>
            )}
          </div>

          {loading ? (
            <div style={{ display:'flex', justifyContent:'center', padding:56 }}><Spinner size={28} /></div>
          ) : history.length === 0 ? (
            <EmptyState />
          ) : (
            <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
              {history.map((s, i) => <SessionRow key={s.id} session={s} index={i} />)}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

/*  StatCard  */
function StatCard({ icon, label, value, visible, delay }) {
  return (
    <div className="stat-card" style={{
      background: T.cardBg,
      backdropFilter: 'blur(12px)',
      border: `1px solid ${T.border}`,
      borderRadius: T.radius,
      padding: '22px 24px',
      animation: visible ? `fadeUp 0.5s ${delay} ease both` : 'none',
      opacity: visible ? undefined : 0,
    }}>
      <div style={{ fontSize:22, marginBottom:12 }}>{icon}</div>
      <div style={{
        fontFamily: 'Lora, serif',
        fontSize: 34,
        fontWeight: 700,
        color: T.ink,
        lineHeight: 1,
        marginBottom: 6,
        animation: visible ? `countUp 0.4s ${delay} ease both` : 'none',
      }}>
        {value}
      </div>
      <div style={{ fontSize:12, color:T.muted, fontWeight:500, letterSpacing:0.3 }}>{label}</div>
    </div>
  )
}

/*  SessionRow  */
function SessionRow({ session: s, index }) {
  const colour = OVERALL_RATING_COLOURS[s.overall_rating] ?? T.muted
  const date   = new Date(s.created_at).toLocaleDateString('en-GB', {
    day:'numeric', month:'short', year:'numeric',
  })

  return (
    <Link to={`/session/${s.id}`} style={{ textDecoration:'none' }}>
      <div className="session-row" style={{
        background: T.cardBg,
        backdropFilter: 'blur(8px)',
        border: `1px solid ${T.border}`,
        borderRadius: 12,
        padding: '14px 18px',
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        cursor: 'pointer',
        animation: `fadeUp 0.4s ${0.05 * index + 0.5}s ease both`,
        opacity: 0,
        animationFillMode: 'forwards',
      }}>
        {/* Icon dot */}
        <div style={{
          width: 38, height: 38, borderRadius: '50%', flexShrink:0,
          background: s.exercise_completed
            ? 'rgba(74,123,95,0.12)'
            : `${T.paper}`,
          border: s.exercise_completed
            ? `1.5px solid rgba(74,123,95,0.25)`
            : `1.5px solid ${T.border}`,
          display:'flex', alignItems:'center', justifyContent:'center',
          fontSize:15,
        }}>
          {s.exercise_completed ? '✓' : '○'}
        </div>

        {/* Text */}
        <div style={{ flex:1, minWidth:0 }}>
          <div style={{
            fontSize:13, fontWeight:600, color:T.ink,
            marginBottom:3, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
          }}>
            {s.recommended_exercise_name ?? 'Assessment only'}
          </div>
          <div style={{ fontSize:11, color:T.muted, fontWeight:400 }}>{date}</div>
        </div>

        {/* Badge */}
        {s.overall_rating && (
          <span className="badge" style={{
            background: `${colour}18`,
            color: colour,
            border: `1px solid ${colour}33`,
          }}>
            {s.overall_rating.replace('_', ' ')}
          </span>
        )}

        <span style={{ fontSize:16, color:T.muted, fontWeight:300 }}>›</span>
      </div>
    </Link>
  )
}

/*  EmptyState  */
function EmptyState() {
  const navigate = useNavigate()
  return (
    <div style={{
      background: T.cardBg,
      border: `1px dashed rgba(30,58,47,0.2)`,
      borderRadius: T.radius,
      padding: '56px 32px',
      textAlign:'center',
    }}>
      <div style={{
        width:56, height:56, borderRadius:'50%',
        background:'rgba(74,123,95,0.1)',
        border:`1.5px solid rgba(74,123,95,0.2)`,
        display:'flex', alignItems:'center', justifyContent:'center',
        fontSize:22, margin:'0 auto 18px',
      }}>
        🎙
      </div>
      <h3 style={{
        fontFamily:'Lora, serif',
        fontSize:18, fontWeight:600,
        color:T.ink, margin:'0 0 8px',
      }}>
        No sessions yet
      </h3>
      <p style={{ color:T.muted, fontSize:14, margin:'0 0 24px', maxWidth:280, marginInline:'auto' }}>
        Start your first session to track your speech progress over time.
      </p>
      <button className="start-btn" onClick={() => navigate('/session')}>
        Begin First Session →
      </button>
    </div>
  )
}

/*  Helpers  */
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
