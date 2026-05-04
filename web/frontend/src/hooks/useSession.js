/**
 * useSession — encapsulates all state and async handlers for the therapy
 * session flow. SessionPage is intentionally thin — just wiring this hook
 * to step components.
 *
 * @param {string|null} initialSessionId
 */

import { useCallback, useEffect, useState } from 'react'
import { assessment, recommendations, sessions } from '../api/client'

function normaliseFeedback(raw) {
  if (!raw) return null
  return {
    feedback:       raw.feedback       ?? raw,
    metrics_before: raw.metrics_before ?? null,
    metrics_after:  raw.metrics_after  ?? null,
    transcript:     raw.transcript     ?? null,
  }
}

export function useSession(initialSessionId = null) {
  const [step,             setStep]            = useState(initialSessionId ? 'recommend' : 'assess')
  const [sessionId,        setSessionId]        = useState(initialSessionId)
  const [assessmentResult, setAssessmentResult] = useState(null)
  const [recommendation,   setRecommendation]   = useState(null)
  const [feedbackResult,   setFeedbackResult]   = useState(null)
  const [loading,          setLoading]          = useState(false)
  const [error,            setError]            = useState('')

  //  Load existing session 
  useEffect(() => {
    if (!initialSessionId) return
    setLoading(true)
    setError('')
    sessions.get(initialSessionId)
      .then(s => {
        setAssessmentResult({ metrics: s.metrics, transcript: s.transcript, session_id: s.id })
        if (s.recommended_exercise_name) {
          setRecommendation({
            exercise:          { name: s.recommended_exercise_name, id: s.recommended_exercise_id },
            explanation:       s.explanation,
            confidence:        s.confidence,
            plain_explanation: s.explanation?.plain_explanation ?? null,
          })
          setStep(s.exercise_completed ? 'feedback' : 'exercise')
          if (s.exercise_completed) setFeedbackResult(normaliseFeedback(s.exercise_feedback))
        }
      })
      .catch(e => setError(`Could not load session: ${e.message ?? 'Unknown error'}`))
      .finally(() => setLoading(false))
  }, [initialSessionId])

  //  Shared async runner 
  const call = useCallback(async (fn) => {
    setLoading(true)
    setError('')
    try {
      return await fn()
    } catch (e) {
      setError(e.message ?? 'Something went wrong. Please try again.')
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  //  Step 1 
  const handleTextAssess = useCallback(async (text) => {
    const fd = new FormData()
    fd.append('text', text)
    const res = await call(() => assessment.fromText(fd))
    if (!res) return
    setAssessmentResult(res)
    setSessionId(res.session_id)
    setStep('recommend')
  }, [call])

  const handleFileAssess = useCallback(async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const res = await call(() => assessment.fromUpload(fd))
    if (!res) return
    setAssessmentResult(res)
    setSessionId(res.session_id)
    setStep('recommend')
  }, [call])

  const handleAudioAssess = useCallback(async (blob) => {
    const fd = new FormData()
    fd.append('file', blob, 'recording.webm')
    const res = await call(() => assessment.fromUpload(fd))
    if (!res) return
    setAssessmentResult(res)
    setSessionId(res.session_id)
    setStep('recommend')
  }, [call])

  //  Step 2 
  const handleGetRecommendation = useCallback(async () => {
    const res = await call(() => recommendations.get(sessionId))
    if (!res) return
    setRecommendation({
      ...res,
      plain_explanation: res.explanation?.plain_explanation ?? res.plain_explanation ?? null,
    })
    setStep('exercise')
  }, [call, sessionId])

  //  Step 3 
  const handleExerciseAudio = useCallback(async (blob) => {
    const fd = new FormData()
    fd.append('audio', blob, 'exercise.webm')
    const res = await call(() => recommendations.submitAudio(sessionId, fd))
    if (!res) return
    setFeedbackResult(normaliseFeedback(res))
    setStep('feedback')
  }, [call, sessionId])

  return {
    step,
    sessionId,
    assessmentResult,
    recommendation,
    feedbackResult,
    loading,
    error,
    setError,
    handleTextAssess,
    handleFileAssess,
    handleAudioAssess,
    handleGetRecommendation,
    handleExerciseAudio,
  }
}
