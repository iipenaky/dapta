import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

const NAV = [
  { path: '/dashboard', icon: '🏠', label: 'Dashboard'     },
  { path: '/session',   icon: '🎙', label: 'New Session'   },
  { path: '/profile',   icon: '👤', label: 'My Profile'    },
]

export default function AppShell() {
  const { user, logout } = useAuth()
  const location         = useLocation()
  const navigate         = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside style={{
        width:      240,
        flexShrink: 0,
        background: 'var(--bg-primary)',
        borderRight: '1px solid var(--border)',
        display:    'flex',
        flexDirection: 'column',
        padding:    '24px 0',
        position:   'sticky',
        top:        0,
        height:     '100vh',
      }}>
        {/* Logo */}
        <div style={{ padding: '0 20px 28px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontSize: 22 }}>🧠</div>
          <div style={{ fontWeight: 800, fontSize: 18, marginTop: 4 }}>DAPTA</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Personalised Aphasia Therapy
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '16px 12px' }}>
          {NAV.map(item => {
            const active = location.pathname.startsWith(item.path) && (item.path !== '/' || location.pathname === '/')
            return (
              <Link
                key={item.path}
                to={item.path}
                style={{
                  display:      'flex',
                  alignItems:   'center',
                  gap:          10,
                  padding:      '10px 12px',
                  borderRadius: 10,
                  marginBottom: 4,
                  fontSize:     14,
                  fontWeight:   active ? 600 : 400,
                  color:        active ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  background:   active ? 'rgba(99,179,237,0.1)' : 'transparent',
                  textDecoration: 'none',
                  transition:   'all 0.15s',
                }}
              >
                <span style={{ fontSize: 17 }}>{item.icon}</span>
                {item.label}
              </Link>
            )
          })}
        </nav>

        {/* User + logout */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 2, color: 'var(--text-primary)' }}>
            {user?.full_name}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
            {user?.email}
          </div>
          <button
            onClick={handleLogout}
            style={{
              background: 'transparent', border: '1px solid var(--border)',
              borderRadius: 8, padding: '7px 14px', cursor: 'pointer',
              color: 'var(--text-secondary)', fontSize: 12,
              fontFamily: 'var(--font-body)', width: '100%',
            }}
          >
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, padding: '40px 48px', overflowY: 'auto' }}>
        <Outlet />
      </main>
    </div>
  )
}
