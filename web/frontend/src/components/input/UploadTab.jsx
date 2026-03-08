import { useRef }             from 'react'
import { ACCEPTED_FILES }     from '../../utils/constants'
import { primaryBtn, hintText } from '../../utils/styles'

export default function UploadTab({ loading, onFile }) {
  const ref = useRef()
  return (
    <div style={{ paddingTop: 24 }}>
      <p style={hintText}>
        Upload a <strong style={{ color: 'var(--text-primary)' }}>.cha</strong> CLAN transcript,
        or an audio file (<strong style={{ color: 'var(--text-primary)' }}>.wav · .mp3 · .mp4 · .m4a</strong>).
      </p>
      <input ref={ref} type="file" accept={ACCEPTED_FILES} style={{ display: 'none' }}
        onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
      <button onClick={() => ref.current.click()} disabled={loading} style={primaryBtn}>
        {loading ? 'Processing…' : '📂 Choose File'}
      </button>
    </div>
  )
}
