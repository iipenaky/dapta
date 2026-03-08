import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider }   from './context/AuthContext'
import ProtectedRoute     from './components/auth/ProtectedRoute'
import AppShell           from './components/layout/AppShell'
import LoginPage          from './pages/LoginPage'
import SignupPage         from './pages/SignupPage'
import DashboardPage      from './pages/DashboardPage'
import SessionPage        from './pages/SessionPage'
import ProfilePage        from './pages/ProfilePage'

/**
 * Root component.
 * AuthProvider wraps the entire tree so any component can access auth state.
 * BrowserRouter enables client-side routing.
 */
export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public routes */}
          <Route path="/login"  element={<LoginPage />}  />
          <Route path="/signup" element={<SignupPage />} />

          {/* Protected routes — redirect to /login if not authenticated */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppShell />}>
              <Route path="/"           element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard"  element={<DashboardPage />} />
              <Route path="/session"    element={<SessionPage />}   />
              <Route path="/session/:id" element={<SessionPage />}  />
              <Route path="/profile"    element={<ProfilePage />}   />
            </Route>
          </Route>

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
