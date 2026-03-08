import { useAudioRecorder } from '../hooks/useAudioRecorder'
import { primaryBtn }       from '../utils/styles'

/**
 * Self-contained audio recorder.
 * Calls onRecorded(blob) when the user submits a recording.
 *
 * @param {{ onRecorded: (blob: Blob) => void, loading: boolean, label?: string }} props
 */
export default function AudioRecorder({ onRecorded, loading, label = 'Record Response' }) {
  const { recording, audioURL, audioBlob, error, elapsedSecs, start, stop, reset } = useAudioRecorder()

  const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

  return (
    <div>
      {error && (
        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 12 }}>{error}</p>
      )}

      {!audioURL && (
        <button
          onClick={recording ? stop : start}
          disabled={loading}
          style={{
            ...primaryBtn,
            background: recording ? 'rgba(252,129,129,0.15)' : 'var(--gradient-2)',
            border:     recording ? '1px solid var(--accent-red)' : 'none',
            color:      recording ? 'var(--accent-red)' : 'white',
          }}
        >
          {recording
            ? `⏹ Stop Recording  ${fmt(elapsedSecs)}`
            : `🎙 ${label}`}
        </button>
      )}

      {audioURL && (
        <div style={{ marginTop: 16 }}>
          <audio controls src={audioURL} style={{ width: '100%', marginBottom: 12 }} />
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => onRecorded(audioBlob)}
              disabled={loading}
              style={{ ...primaryBtn, flex: 1 }}
            >
              {loading ? 'Analysing…' : 'Submit Recording →'}
            </button>
            <button
              onClick={reset}
              disabled={loading}
              style={{
                padding: '13px 18px', border: '1px solid var(--border)',
                borderRadius: 10, background: 'transparent',
                color: 'var(--text-secondary)', cursor: 'pointer',
                fontFamily: 'var(--font-body)', fontSize: 13,
              }}
            >
              Re-record
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
