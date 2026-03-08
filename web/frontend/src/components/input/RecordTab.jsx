import AudioRecorder         from '../AudioRecorder'
import { hintText }          from '../../utils/styles'

export default function RecordTab({ loading, onAudio }) {
  return (
    <div style={{ paddingTop: 24 }}>
      <p style={hintText}>
        Press <strong style={{ color: 'var(--text-primary)' }}>Start Recording</strong> and talk
        naturally for 30–60 seconds. Describe a picture, tell a story, or talk about your day.
      </p>
      <AudioRecorder onRecorded={onAudio} loading={loading} label="Start Recording" />
    </div>
  )
}
