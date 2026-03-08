/**
 * useAudioRecorder — MediaRecorder abstraction.
 * Returns controls and state for recording audio in the browser.
 */

import { useCallback, useRef, useState } from 'react'

export function useAudioRecorder() {
  const [recording,    setRecording]    = useState(false)
  const [audioURL,     setAudioURL]     = useState(null)
  const [audioBlob,    setAudioBlob]    = useState(null)
  const [error,        setError]        = useState('')
  const [elapsedSecs,  setElapsedSecs]  = useState(0)

  const mediaRecorderRef = useRef(null)
  const chunksRef        = useRef([])
  const timerRef         = useRef(null)

  const start = useCallback(async () => {
    setError('')
    setAudioURL(null)
    setAudioBlob(null)
    setElapsedSecs(0)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        setAudioURL(URL.createObjectURL(blob))
        stream.getTracks().forEach(t => t.stop())
      }

      recorder.start(250)   // Collect data every 250ms
      setRecording(true)

      timerRef.current = setInterval(() => {
        setElapsedSecs(s => s + 1)
      }, 1000)
    } catch (e) {
      setError('Microphone access denied. Please allow microphone access and try again.')
    }
  }, [])

  const stop = useCallback(() => {
    mediaRecorderRef.current?.stop()
    clearInterval(timerRef.current)
    setRecording(false)
  }, [])

  const reset = useCallback(() => {
    stop()
    setAudioURL(null)
    setAudioBlob(null)
    setElapsedSecs(0)
    setError('')
  }, [stop])

  return { recording, audioURL, audioBlob, error, elapsedSecs, start, stop, reset }
}
