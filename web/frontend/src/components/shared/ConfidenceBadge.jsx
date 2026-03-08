import { CONFIDENCE_COLOURS } from '../../utils/constants'

export default function ConfidenceBadge({ level }) {
  const c = CONFIDENCE_COLOURS[level] ?? 'var(--text-muted)'
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: '3px 10px', borderRadius: 20,
      background: `${c}22`, color: c, border: `1px solid ${c}44`,
      textTransform: 'uppercase', letterSpacing: 0.5,
    }}>
      {level ?? 'Moderate'} Confidence
    </span>
  )
}
