/**
 * @fileoverview API client — all fetch calls to the DAPTA backend.
 *
 * Rules:
 *  - Every function returns the parsed JSON body on success.
 *  - Every function throws an Error with a human-readable message on failure.
 *  - Tokens are stored in localStorage and attached automatically.
 *  - A 401 response triggers a token refresh attempt; if that fails, the user
 *    is signed out.
 */

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// ── Token storage ─────────────────────────────────────────────────────────────

export const tokens = {
  get access()  { return localStorage.getItem('access_token')  },
  get refresh() { return localStorage.getItem('refresh_token') },
  set(access, refresh) {
    localStorage.setItem('access_token',  access)
    localStorage.setItem('refresh_token', refresh)
  },
  clear() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  },
}

// ── Core fetch wrapper ────────────────────────────────────────────────────────

let _isRefreshing    = false
let _refreshQueue    = []   // Pending requests waiting for token refresh

/**
 * Fetch wrapper that:
 *  1. Attaches Authorization header
 *  2. On 401, attempts a token refresh once
 *  3. On refresh failure, clears tokens and reloads
 *
 * @param {string}  path     - API path, e.g. '/api/auth/me'
 * @param {object}  [opts]   - fetch options
 * @param {boolean} [retry]  - internal flag to prevent infinite retry loop
 */
async function request(path, opts = {}, retry = true) {
  const headers = { ...(opts.headers ?? {}) }

  // Don't set Content-Type for FormData — browser sets it with boundary
  if (!(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  if (tokens.access) {
    headers['Authorization'] = `Bearer ${tokens.access}`
  }

  const res = await fetch(`${BASE}${path}`, { ...opts, headers })

  // ── 401 handling ──────────────────────────────────────────────────────────
  if (res.status === 401 && retry && tokens.refresh) {
    if (_isRefreshing) {
      // Queue this request until the in-flight refresh completes
      return new Promise((resolve, reject) => {
        _refreshQueue.push({ resolve, reject, path, opts })
      })
    }

    _isRefreshing = true
    try {
      const refreshed = await request('/api/auth/refresh', {
        method: 'POST',
        body: JSON.stringify({ refresh_token: tokens.refresh }),
      }, false)

      tokens.set(refreshed.access_token, refreshed.refresh_token)

      // Drain queue
      _refreshQueue.forEach(({ resolve, reject, path: p, opts: o }) =>
        request(p, o, false).then(resolve).catch(reject)
      )
      _refreshQueue = []

      // Retry original request
      return request(path, opts, false)
    } catch {
      tokens.clear()
      window.location.href = '/login'
      throw new Error('Session expired. Please log in again.')
    } finally {
      _isRefreshing = false
    }
  }

  // ── Parse body ────────────────────────────────────────────────────────────
  if (res.status === 204) return null

  const body = await res.json().catch(() => ({ detail: res.statusText }))

  if (!res.ok) {
    throw new Error(body.detail ?? `Request failed (${res.status})`)
  }

  return body
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export const auth = {
  /** @param {{ full_name: string, email: string, password: string }} data */
  signup: (data) => request('/api/auth/signup', {
    method: 'POST', body: JSON.stringify(data),
  }),

  /** @param {{ email: string, password: string }} data */
  login: (data) => request('/api/auth/login', {
    method: 'POST', body: JSON.stringify(data),
  }),

  /** @param {string} refresh_token */
  refresh: (refresh_token) => request('/api/auth/refresh', {
    method: 'POST', body: JSON.stringify({ refresh_token }),
  }),

  me: () => request('/api/auth/me'),

  /** @param {object} data */
  updateProfile: (data) => request('/api/auth/me', {
    method: 'PATCH', body: JSON.stringify(data),
  }),

  logout: () => request('/api/auth/logout', { method: 'POST' }),
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export const sessions = {
  list: (limit = 20, offset = 0) =>
    request(`/api/sessions/?limit=${limit}&offset=${offset}`),

  get: (id) => request(`/api/sessions/${id}`),

  delete: (id) => request(`/api/sessions/${id}`, { method: 'DELETE' }),
}

// ── Assessment ────────────────────────────────────────────────────────────────

export const assessment = {
  /** @param {FormData} fd — must contain `text` field */
  fromText: (fd) => request('/api/assessment/text', { method: 'POST', body: fd }),

  /** @param {FormData} fd — must contain `file` field */
  fromUpload: (fd) => request('/api/assessment/upload', { method: 'POST', body: fd }),
}

// ── Recommendations ───────────────────────────────────────────────────────────

export const recommendations = {
  /** @param {string} sessionId */
  get: (sessionId) => request(`/api/recommendations/${sessionId}`),

  /**
   * @param {string}   sessionId
   * @param {FormData} fd — must contain `audio` field (Blob)
   */
  submitAudio: (sessionId, fd) => request(
    `/api/recommendations/${sessionId}/submit-audio`,
    { method: 'POST', body: fd },
  ),
}
