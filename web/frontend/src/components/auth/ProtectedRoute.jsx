import { Navigate, Outlet } from 'react-router-dom'
import { useAuth }           from '../../context/AuthContext'
import Spinner               from '../shared/Spinner'

/**
 * Renders child routes only when authenticated.
 * Shows a spinner while the initial /me request is in-flight.
 */
export default function ProtectedRoute() {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Spinner size={32} />
      </div>
    )
  }

  return user ? <Outlet /> : <Navigate to="/login" replace />
}
