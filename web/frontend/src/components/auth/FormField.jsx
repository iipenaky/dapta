import { inputStyle, labelStyle } from '../../utils/styles'

/**
 * Labelled form field with optional error message.
 * @param {{ label, id, error, ...inputProps }} props
 */
export default function FormField({ label, id, error, ...inputProps }) {
  return (
    <div>
      <label htmlFor={id} style={labelStyle}>{label}</label>
      <input
        id={id}
        {...inputProps}
        style={{
          ...inputStyle,
          borderColor: error ? 'var(--accent-red)' : undefined,
        }}
      />
      {error && (
        <p style={{ fontSize: 12, color: 'var(--accent-red)', marginTop: 4 }}>{error}</p>
      )}
    </div>
  )
}
