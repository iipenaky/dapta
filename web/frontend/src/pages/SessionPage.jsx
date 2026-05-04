import { useNavigate, useParams } from 'react-router-dom'
import { useSession }              from '../hooks/useSession'
import AssessStep                  from '../components/session/AssessStep'
import RecommendStep               from '../components/session/RecommendStep'
import ExerciseStep                from '../components/session/ExerciseStep'
import FeedbackStep                from '../components/session/FeedbackStep'
import ErrorBanner                 from '../components/shared/ErrorBanner'

/*  Google Fonts  */
const fontLink = document.createElement('link')
fontLink.rel  = 'stylesheet'
fontLink.href = 'https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&display=swap'
if (!document.head.querySelector('link[href*="Lora"]')) document.head.appendChild(fontLink)

/*  Design tokens  */
const T = {
  cream:  '#F5F0E8',
  paper:  '#EDE8DC',
  forest: '#1E3A2F',
  sage:   '#4A7B5F',
  moss:   '#6B9E7E',
  sand:   '#C4A882',
  amber:  '#D4843A',
  ink:    '#1A1A1A',
  muted:  '#7A7268',
  border: 'rgba(30,58,47,0.12)',
  cardBg: 'rgba(255,255,255,0.72)',
}

/*  Step config  */
const STEPS = [
  { key: 'assess',    label: 'Assess',      icon: '🎙' },
  { key: 'recommend', label: 'Recommend',   icon: '🧠' },
  { key: 'exercise',  label: 'Exercise',    icon: '💬' },
  { key: 'feedback',  label: 'Feedback',    icon: '📊' },
]

const css = `
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes waveIn {
    from { transform:scaleX(0); }
    to   { transform:scaleX(1); }
  }
  @keyframes pulse {
    0%, 100% { opacity:1; }
    50%       { opacity:0.5; }
  }
  .session-root * { font-family:'DM Sans', sans-serif; box-sizing:border-box; }
  .step-dot {
    transition: all 0.25s ease;
  }
  .step-dot.active {
    background: ${T.forest} !important;
    border-color: ${T.forest} !important;
    color: #fff !important;
    transform: scale(1.08);
    box-shadow: 0 4px 16px rgba(30,58,47,0.25);
  }
  .step-dot.done {
    background: ${T.sage} !important;
    border-color: ${T.sage} !important;
    color: #fff !important;
  }
  .step-dot.upcoming {
    background: transparent !important;
    border-color: rgba(30,58,47,0.2) !important;
    color: ${T.muted} !important;
  }
  .content-card {
    background: ${T.cardBg};
    backdrop-filter: blur(12px);
    border: 1px solid ${T.border};
    border-radius: 18px;
    padding: 36px;
    animation: fadeUp 0.45s ease both;
  }
`

export default function SessionPage() {
  const { id }   = useParams()
  const navigate = useNavigate()
  const session  = useSession(id ?? null)

  const {
    step, assessmentResult, recommendation, feedbackResult,
    loading, error, setError,
    handleTextAssess, handleFileAssess, handleAudioAssess,
    handleGetRecommendation, handleExerciseAudio,
  } = session

  const stepIndex = STEPS.findIndex(s => s.key === step)

  return (
    <div className="session-root" style={{
      minHeight: '100vh',
      background: T.cream,
      backgroundImage: `
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(74,123,95,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 90%, rgba(212,132,58,0.06) 0%, transparent 55%)
      `,
    }}>
      <style>{css}</style>

      <div style={{ maxWidth: 720, margin: '0 auto', padding: '48px 24px 80px' }}>

        {/*  Header  */}
        <header style={{ marginBottom: 40, animation: 'fadeUp 0.5s ease both' }}>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:10 }}>
            <span style={{
              display:'inline-block', width:28, height:3,
              background: T.sage, borderRadius:2,
              transformOrigin:'left',
              animation: 'waveIn 0.6s 0.2s ease both',
            }} />
            <span style={{
              fontSize:12, fontWeight:600, color:T.sage,
              letterSpacing:1.8, textTransform:'uppercase',
            }}>
              Therapy Session
            </span>
          </div>
          <h1 style={{
            fontFamily: 'Lora, serif',
            fontSize: 30,
            fontWeight: 700,
            color: T.ink,
            margin: 0,
            lineHeight: 1.25,
            letterSpacing: '-0.5px',
          }}>
            {STEPS[stepIndex]?.icon} {stepLabel(step)}
          </h1>
          <p style={{ marginTop:6, color:T.muted, fontSize:14, fontWeight:400 }}>
            {stepSubtitle(step)}
          </p>
        </header>

        {/*  Step indicator  */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          marginBottom: 36,
          animation: 'fadeUp 0.5s 0.1s ease both',
          opacity: 0,
          animationFillMode: 'forwards',
        }}>
          {STEPS.map((s, i) => {
            const state = i < stepIndex ? 'done' : i === stepIndex ? 'active' : 'upcoming'
            return (
              <div key={s.key} style={{ display:'flex', alignItems:'center', flex: i < STEPS.length - 1 ? 1 : 'none' }}>
                {/* Dot */}
                <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:6 }}>
                  <div className={`step-dot ${state}`} style={{
                    width: 36, height: 36,
                    borderRadius: '50%',
                    border: '2px solid',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: state === 'done' ? 14 : 13,
                    fontWeight: 600,
                    cursor: 'default',
                  }}>
                    {state === 'done' ? '✓' : s.icon}
                  </div>
                  <span style={{
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: 0.8,
                    textTransform: 'uppercase',
                    color: state === 'upcoming' ? T.muted : state === 'active' ? T.forest : T.sage,
                  }}>
                    {s.label}
                  </span>
                </div>

                {/* Connector line */}
                {i < STEPS.length - 1 && (
                  <div style={{
                    flex: 1,
                    height: 2,
                    marginBottom: 22,
                    marginInline: 8,
                    background: i < stepIndex
                      ? T.sage
                      : 'rgba(30,58,47,0.12)',
                    borderRadius: 2,
                    transition: 'background 0.4s ease',
                  }} />
                )}
              </div>
            )
          })}
        </div>

        {/*  Error banner  */}
        {error && (
          <div style={{
            background: 'rgba(212,58,58,0.08)',
            border: '1px solid rgba(212,58,58,0.2)',
            borderRadius: 12,
            padding: '12px 16px',
            marginBottom: 20,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: 13,
            color: '#B03030',
            fontWeight: 500,
          }}>
            <span>⚠️ {error}</span>
            <button onClick={() => setError('')} style={{
              background: 'none', border: 'none',
              cursor: 'pointer', color: '#B03030',
              fontSize: 16, lineHeight: 1, padding: '0 4px',
            }}>×</button>
          </div>
        )}

        {/*  Loading overlay  */}
        {loading && (
          <div style={{
            background: T.cardBg,
            backdropFilter: 'blur(12px)',
            border: `1px solid ${T.border}`,
            borderRadius: 18,
            padding: '48px 36px',
            textAlign: 'center',
            marginBottom: 20,
            animation: 'fadeUp 0.3s ease both',
          }}>
            <div style={{
              width: 48, height: 48,
              borderRadius: '50%',
              border: `3px solid rgba(30,58,47,0.12)`,
              borderTopColor: T.sage,
              margin: '0 auto 16px',
              animation: 'spin 0.9s linear infinite',
            }} />
            <style>{`@keyframes spin { to { transform:rotate(360deg); } }`}</style>
            <p style={{ color: T.muted, fontSize: 14, margin: 0 }}>
              {loadingMessage(step)}
            </p>
          </div>
        )}

        {/*  Step content  */}
        {!loading && (
          <div className="content-card">
            {step === 'assess' && (
              <AssessStep
                loading={loading}
                onText={handleTextAssess}
                onFile={handleFileAssess}
                onAudio={handleAudioAssess}
              />
            )}
            {step === 'recommend' && (
              <RecommendStep
                result={assessmentResult}
                loading={loading}
                onGetRec={handleGetRecommendation}
              />
            )}
            {step === 'exercise' && (
              <ExerciseStep
                recommendation={recommendation}
                loading={loading}
                onSubmitAudio={handleExerciseAudio}
              />
            )}
            {step === 'feedback' && (
              <FeedbackStep
                feedback={feedbackResult}
                onNewSession={() => navigate('/session')}
                onDashboard={() => navigate('/dashboard')}
              />
            )}
          </div>
        )}

        {/*  Back to dashboard  */}
        <div style={{ marginTop: 28, textAlign: 'center' }}>
          <button onClick={() => navigate('/dashboard')} style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: T.muted,
            fontSize: 13,
            fontFamily: 'DM Sans, sans-serif',
            fontWeight: 500,
            textDecoration: 'underline',
            textUnderlineOffset: 3,
          }}>
            ← Back to dashboard
          </button>
        </div>

      </div>
    </div>
  )
}

/*  Helpers  */
function stepLabel(step) {
  return {
    assess:    'Speech Assessment',
    recommend: 'Your Recommendation',
    exercise:  'Practice Exercise',
    feedback:  'Session Feedback',
  }[step] ?? 'Session'
}

function stepSubtitle(step) {
  return {
    assess:    'Provide a speech sample so we can assess your current profile.',
    recommend: 'Review your results and get a personalised exercise recommendation.',
    exercise:  'Complete the recommended exercise at your own pace.',
    feedback:  'Here is how you did — keep up the great work.',
  }[step] ?? ''
}

function loadingMessage(step) {
  return {
    assess:    'Analysing your speech sample…',
    recommend: 'Generating your personalised recommendation…',
    exercise:  'Processing your exercise response…',
    feedback:  'Preparing your feedback…',
  }[step] ?? 'Loading…'
}