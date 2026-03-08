/**
 * @fileoverview Shared inline style objects.
 * CSS-only — no data, no logic.
 */

export const primaryBtn = {
  padding:      '13px 24px',
  border:       'none',
  borderRadius: 10,
  background:   'var(--gradient-2)',
  color:        'white',
  fontFamily:   'var(--font-body)',
  fontSize:     14,
  fontWeight:   600,
  cursor:       'pointer',
  transition:   'opacity 0.2s',
  width:        '100%',
}

export const secondaryBtn = {
  padding:      '13px 24px',
  borderRadius: 10,
  border:       '1px solid var(--border)',
  background:   'transparent',
  color:        'var(--text-secondary)',
  fontFamily:   'var(--font-body)',
  fontSize:     14,
  fontWeight:   500,
  cursor:       'pointer',
}

export const sectionTitle = {
  fontSize:      12,
  fontWeight:    600,
  color:         'var(--text-secondary)',
  textTransform: 'uppercase',
  letterSpacing: 0.8,
  marginBottom:  12,
}

export const hintText = {
  color:        'var(--text-secondary)',
  fontSize:     13,
  marginBottom: 20,
  lineHeight:   1.6,
}

export const inputStyle = {
  width:        '100%',
  padding:      '12px 14px',
  borderRadius: 10,
  background:   'var(--bg-primary)',
  border:       '1px solid var(--border)',
  color:        'var(--text-primary)',
  fontFamily:   'var(--font-body)',
  fontSize:     14,
  outline:      'none',
  lineHeight:   1.5,
}

export const labelStyle = {
  display:      'block',
  fontSize:     13,
  fontWeight:   500,
  color:        'var(--text-secondary)',
  marginBottom: 6,
}
