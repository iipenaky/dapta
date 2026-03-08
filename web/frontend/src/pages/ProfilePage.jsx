import { useState }    from 'react'
import { useAuth }     from '../context/AuthContext'
import { APHASIA_SUBTYPES } from '../utils/constants'
import { primaryBtn, inputStyle, labelStyle } from '../utils/styles'

export default function ProfilePage() {
  const { user, updateProfile } = useAuth()
  const [form, setForm]         = useState({
    full_name:        user?.full_name        ?? '',
    aphasia_subtype:  user?.aphasia_subtype  ?? '',
    wab_aq:           user?.wab_aq           ?? '',
    months_post_onset: user?.months_post_onset ?? '',
  })
  const [saving,   setSaving]   = useState(false)
  const [success,  setSuccess]  = useState(false)
  const [error,    setError]    = useState('')

  const set = (k) => (e) => {
    setSuccess(false)
    setForm(f => ({ ...f, [k]: e.target.value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError('')
    setSuccess(false)
    try {
      const payload = {
        full_name:         form.full_name || undefined,
        aphasia_subtype:   form.aphasia_subtype || undefined,
        wab_aq:            form.wab_aq !== '' ? Number(form.wab_aq) : undefined,
        months_post_onset: form.months_post_onset !== '' ? Number(form.months_post_onset) : undefined,
      }
      await updateProfile(payload)
      setSuccess(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={{ maxWidth: 560, margin: '0 auto' }}>
      <h1 style={{ fontSize: 26, fontWeight: 800, marginBottom: 6 }}>My Profile</h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 32 }}>
        Your clinical profile helps DAPTA personalise your therapy sessions.
      </p>

      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)', padding: '32px',
      }}>
        {/* Account info (read-only) */}
        <section style={{ marginBottom: 28, paddingBottom: 28, borderBottom: '1px solid var(--border)' }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 16 }}>
            Account
          </h3>
          <div style={{ fontSize: 14, color: 'var(--text-primary)', marginBottom: 6 }}>
            <span style={{ color: 'var(--text-muted)', marginRight: 8 }}>Email</span>
            {user?.email}
          </div>
        </section>

        {/* Editable fields */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 0.8, margin: 0 }}>
            Clinical Profile
          </h3>

          {/* Full name */}
          <div>
            <label style={labelStyle}>Full name</label>
            <input type="text" value={form.full_name} onChange={set('full_name')} style={inputStyle} />
          </div>

          {/* Aphasia subtype */}
          <div>
            <label style={labelStyle}>Aphasia subtype</label>
            <select value={form.aphasia_subtype} onChange={set('aphasia_subtype')} style={{ ...inputStyle, appearance: 'none' }}>
              <option value="">Not set</option>
              {APHASIA_SUBTYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              Ask your speech-language pathologist if unsure.
            </p>
          </div>

          {/* WAB-AQ */}
          <div>
            <label style={labelStyle}>WAB Aphasia Quotient (0 – 100)</label>
            <input
              type="number" min="0" max="100" step="0.1"
              value={form.wab_aq} onChange={set('wab_aq')}
              placeholder="e.g. 63.5" style={inputStyle}
            />
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              Your most recent WAB-AQ score from formal assessment.
            </p>
          </div>

          {/* Months post-onset */}
          <div>
            <label style={labelStyle}>Months since onset</label>
            <input
              type="number" min="0" step="1"
              value={form.months_post_onset} onChange={set('months_post_onset')}
              placeholder="e.g. 18" style={inputStyle}
            />
          </div>

          {/* Feedback */}
          {error && (
            <p style={{ fontSize: 13, color: 'var(--accent-red)' }}>⚠️ {error}</p>
          )}
          {success && (
            <p style={{ fontSize: 13, color: 'var(--accent-green)' }}>✓ Profile saved successfully.</p>
          )}

          <button type="submit" disabled={saving} style={primaryBtn}>
            {saving ? 'Saving…' : 'Save Changes'}
          </button>
        </form>
      </div>
    </div>
  )
}
