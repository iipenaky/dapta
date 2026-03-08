import { useState }          from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth }           from '../context/AuthContext'
import AuthForm              from '../components/auth/AuthForm'
import FormField             from '../components/auth/FormField'
import { primaryBtn }        from '../utils/styles'

export default function SignupPage() {
  const { signup }  = useAuth()
  const navigate    = useNavigate()
  const [form, setForm]     = useState({ full_name: '', email: '', password: '', confirm: '' })
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState('')

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  const validate = () => {
    const e = {}
    if (form.full_name.trim().length < 2)   e.full_name = 'Name must be at least 2 characters.'
    if (!form.email.includes('@'))           e.email     = 'Enter a valid email address.'
    if (form.password.length < 8)            e.password  = 'Password must be at least 8 characters.'
    if (!/\d/.test(form.password))           e.password  = 'Password must contain a number.'
    if (!/[A-Z]/.test(form.password))        e.password  = 'Password must contain an uppercase letter.'
    if (form.password !== form.confirm)      e.confirm   = 'Passwords do not match.'
    return e
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setApiError('')
    const errs = validate()
    if (Object.keys(errs).length) { setErrors(errs); return }
    setErrors({})
    setLoading(true)
    try {
      await signup(form.full_name, form.email, form.password)
      navigate('/dashboard')
    } catch (err) {
      setApiError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthForm
      title="Create your account"
      subtitle="Join DAPTA and start your personalised therapy journey."
      error={apiError}
      onSubmit={handleSubmit}
    >
      <FormField label="Full name" id="full_name" type="text"
        value={form.full_name} onChange={set('full_name')}
        error={errors.full_name} autoComplete="name" required />

      <FormField label="Email address" id="email" type="email"
        value={form.email} onChange={set('email')}
        error={errors.email} autoComplete="email" required />

      <FormField label="Password" id="password" type="password"
        value={form.password} onChange={set('password')}
        error={errors.password} autoComplete="new-password" required />

      <FormField label="Confirm password" id="confirm" type="password"
        value={form.confirm} onChange={set('confirm')}
        error={errors.confirm} autoComplete="new-password" required />

      <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Password must be at least 8 characters and contain one number and one uppercase letter.
      </p>

      <button type="submit" disabled={loading} style={primaryBtn}>
        {loading ? 'Creating account…' : 'Create Account'}
      </button>

      <p style={{ textAlign: 'center', fontSize: 13, color: 'var(--text-secondary)' }}>
        Already have an account?{' '}
        <Link to="/login" style={{ color: 'var(--accent-blue)' }}>Sign in</Link>
      </p>
    </AuthForm>
  )
}
