import { useState }        from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth }          from '../context/AuthContext'
import AuthForm             from '../components/auth/AuthForm'
import FormField            from '../components/auth/FormField'
import { primaryBtn }       from '../utils/styles'

export default function LoginPage() {
  const { login }   = useAuth()
  const navigate    = useNavigate()
  const [form, setForm]   = useState({ email: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(form.email, form.password)
      navigate('/dashboard')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthForm
      title="Welcome back"
      subtitle="Sign in to continue your therapy sessions."
      error={error}
      onSubmit={handleSubmit}
    >
      <FormField
        label="Email address" id="email" type="email"
        value={form.email} onChange={set('email')}
        autoComplete="email" required
      />
      <FormField
        label="Password" id="password" type="password"
        value={form.password} onChange={set('password')}
        autoComplete="current-password" required
      />
      <button type="submit" disabled={loading} style={primaryBtn}>
        {loading ? 'Signing in…' : 'Sign In'}
      </button>
      <p style={{ textAlign: 'center', fontSize: 13, color: 'var(--text-secondary)' }}>
        No account?{' '}
        <Link to="/signup" style={{ color: 'var(--accent-blue)' }}>Create one</Link>
      </p>
    </AuthForm>
  )
}
