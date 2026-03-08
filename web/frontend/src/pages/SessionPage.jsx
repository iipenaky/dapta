import { useNavigate, useParams } from 'react-router-dom'
import { useSession }              from '../hooks/useSession'
import StepIndicator               from '../components/session/StepIndicator'
import AssessStep                  from '../components/session/AssessStep'
import RecommendStep               from '../components/session/RecommendStep'
import ExerciseStep                from '../components/session/ExerciseStep'
import FeedbackStep                from '../components/session/FeedbackStep'
import ErrorBanner                 from '../components/shared/ErrorBanner'

/**
 * Orchestrates the 4-step therapy session flow.
 * All state lives in useSession — this component is pure wiring.
 */
export default function SessionPage() {
  const { id }     = useParams()
  const navigate   = useNavigate()
  const session    = useSession(id ?? null)

  const {
    step, assessmentResult, recommendation, feedbackResult,
    loading, error, setError,
    handleTextAssess, handleFileAssess, handleAudioAssess,
    handleGetRecommendation, handleExerciseAudio,
  } = session

  return (
    <div style={{ maxWidth: 700, margin: '0 auto' }}>
      <StepIndicator step={step} />

      {error && <ErrorBanner msg={error} onClose={() => setError('')} />}

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
  )
}
