import { Card, ConfidenceBadge, SkeletonLoader } from '../shared'
import AudioRecorder                              from '../AudioRecorder'

export default function ExerciseStep({ recommendation, loading, onSubmitAudio }) {
  const ex = recommendation?.exercise ?? {}
  return (
    <Card title="Your Exercise" subtitle="Complete this exercise and record your response.">
      {/* Exercise card */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(99,179,237,0.1) 0%, rgba(79,209,197,0.06) 100%)',
        border: '1px solid var(--border-accent)', borderRadius: 12, padding: '20px 24px', marginBottom: 24,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <h3 style={{ fontSize: 20, color: 'var(--accent-blue)', fontWeight: 700 }}>
            {ex.name?.replace(/_/g, ' ') ?? 'Therapy Exercise'}
          </h3>
          <ConfidenceBadge level={recommendation?.confidence} />
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}>
          {ex.description ?? 'Follow the exercise instructions.'}
        </p>
        {ex.context && (
          <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: '12px 14px', fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6 }}>
            <strong style={{ color: 'var(--accent-teal)' }}>How to do it: </strong>{ex.context}
          </div>
        )}
      </div>

      {/* XAI explanation */}
      {recommendation?.plain_explanation && (
        <div style={{
          border: '1px solid var(--border)', borderRadius: 10, padding: '14px 18px',
          marginBottom: 24, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7,
        }}>
          <span style={{ fontSize: 16, marginRight: 8 }}>💡</span>
          {recommendation.plain_explanation}
        </div>
      )}

      {loading ? (
        <SkeletonLoader rows={3} message="Transcribing and analysing your response…" />
      ) : (
        <>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.6 }}>
            Press <strong style={{ color: 'var(--text-primary)' }}>Start Recording</strong> and
            complete the exercise. Speak clearly and take your time — there is no rush.
          </p>
          <AudioRecorder onRecorded={onSubmitAudio} loading={loading} label="Record Your Response" />
        </>
      )}
    </Card>
  )
}
