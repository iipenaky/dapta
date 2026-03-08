/**
 * AuthContext — global authentication state.
 *
 * Provides: { user, loading, login, signup, logout, updateProfile }
 * to any component in the tree via useAuth().
 */

import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { auth, tokens } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user,    setUser]    = useState(null)
  const [loading, setLoading] = useState(true)   // true until initial /me resolves

  // ── Bootstrap: restore session on page load ───────────────────────────────
  useEffect(() => {
    if (!tokens.access) {
      setLoading(false)
      return
    }
    auth.me()
      .then(setUser)
      .catch(() => tokens.clear())
      .finally(() => setLoading(false))
  }, [])

  // ── Actions ───────────────────────────────────────────────────────────────

  const login = useCallback(async (email, password) => {
    const res = await auth.login({ email, password })
    tokens.set(res.access_token, res.refresh_token)
    const me = await auth.me()
    setUser(me)
    return me
  }, [])

  const signup = useCallback(async (full_name, email, password) => {
    const res = await auth.signup({ full_name, email, password })
    tokens.set(res.access_token, res.refresh_token)
    const me = await auth.me()
    setUser(me)
    return me
  }, [])

  const logout = useCallback(async () => {
    try { await auth.logout() } catch { /* ignore */ }
    tokens.clear()
    setUser(null)
  }, [])

  const updateProfile = useCallback(async (data) => {
    const updated = await auth.updateProfile(data)
    setUser(updated)
    return updated
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout, updateProfile }}>
      {children}
    </AuthContext.Provider>
  )
}

/** @returns {{ user, loading, login, signup, logout, updateProfile }} */
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
