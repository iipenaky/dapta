import { useState }              from 'react'
import { primaryBtn, hintText, inputStyle } from '../../utils/styles'

export default function TypeTab({ loading, onText }) {
  const [text, setText] = useState('')
  return (
    <div style={{ paddingTop: 24 }}>
      <p style={hintText}>Type or paste a speech sample — a few sentences is enough.</p>
      <textarea
        value={text} onChange={e => setText(e.target.value)} rows={5}
        placeholder="The woman is washing the dishes. The water is overflowing…"
        style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.6 }}
      />
      <button onClick={() => onText(text)} disabled={loading || !text.trim()}
        style={{ ...primaryBtn, marginTop: 12 }}>
        {loading ? 'Analysing…' : 'Analyse Text'}
      </button>
    </div>
  )
}
