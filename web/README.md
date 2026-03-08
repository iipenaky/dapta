# DAPTA Web — Full-Stack Application

Discourse-Aware Personalised Therapy Agent — web interface with authentication, session management, and the full 4-step therapy flow.

---

## Project Structure

```
web/
├── api/                        # FastAPI backend
│   ├── core/
│   │   ├── config.py           # All env vars / settings (pydantic-settings)
│   │   ├── security.py         # bcrypt hashing + JWT create/verify
│   │   ├── dependencies.py     # FastAPI Depends() providers
│   │   └── middleware.py       # Global error handler middleware
│   ├── models/
│   │   ├── user.py             # User ORM model (SQLAlchemy)
│   │   └── session.py          # Session ORM model
│   ├── schemas/
│   │   ├── auth.py             # Pydantic request/response shapes for auth
│   │   └── session.py          # Shapes for assessment, recommendation, feedback
│   ├── services/
│   │   ├── user_service.py     # All DB operations for users
│   │   ├── session_service.py  # All DB operations for sessions
│   │   └── dapta_service.py    # DAPTA ML pipeline wrapper (singleton)
│   ├── routers/
│   │   ├── auth.py             # POST /signup /login /refresh /logout, GET/PATCH /me
│   │   ├── sessions.py         # GET / (list), GET /:id, DELETE /:id
│   │   ├── assessment.py       # POST /text, POST /upload
│   │   └── recommendations.py  # GET /:sessionId, POST /:sessionId/submit-audio
│   ├── database.py             # Async SQLAlchemy engine + Base
│   ├── main.py                 # App factory, lifespan, router wiring
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/                   # React + Vite
    ├── src/
    │   ├── api/
    │   │   └── client.js       # All fetch calls, token storage, auto-refresh
    │   ├── context/
    │   │   └── AuthContext.jsx # Global auth state + login/signup/logout actions
    │   ├── hooks/
    │   │   ├── useSession.js   # Full 4-step therapy session state machine
    │   │   └── useAudioRecorder.js  # MediaRecorder abstraction
    │   ├── pages/
    │   │   ├── LoginPage.jsx
    │   │   ├── SignupPage.jsx
    │   │   ├── DashboardPage.jsx
    │   │   ├── SessionPage.jsx
    │   │   └── ProfilePage.jsx
    │   ├── components/
    │   │   ├── auth/
    │   │   │   ├── ProtectedRoute.jsx   # Redirects to /login if not authed
    │   │   │   ├── AuthForm.jsx         # Card shell for login/signup
    │   │   │   └── FormField.jsx        # Labelled input with error state
    │   │   ├── layout/
    │   │   │   └── AppShell.jsx         # Sidebar nav + <Outlet />
    │   │   ├── session/
    │   │   │   ├── StepIndicator.jsx
    │   │   │   ├── AssessStep.jsx
    │   │   │   ├── RecommendStep.jsx
    │   │   │   ├── ExerciseStep.jsx
    │   │   │   └── FeedbackStep.jsx
    │   │   ├── input/
    │   │   │   ├── RecordTab.jsx
    │   │   │   ├── UploadTab.jsx
    │   │   │   └── TypeTab.jsx
    │   │   ├── shared/
    │   │   │   ├── Card.jsx
    │   │   │   ├── TabBar.jsx
    │   │   │   ├── MetricBar.jsx
    │   │   │   ├── ConfidenceBadge.jsx
    │   │   │   ├── ErrorBanner.jsx
    │   │   │   ├── SkeletonLoader.jsx
    │   │   │   └── Spinner.jsx
    │   │   └── AudioRecorder.jsx
    │   ├── utils/
    │   │   ├── constants.js     # METRIC_LIST, APHASIA_SUBTYPES, etc.
    │   │   └── styles.js        # Shared inline style objects
    │   ├── styles/
    │   │   └── global.css       # Design tokens, reset, animations
    │   ├── App.jsx              # Router + AuthProvider
    │   └── main.jsx             # React DOM entry
    ├── index.html
    ├── vite.config.js
    └── package.json
```

---

## Setup

### 1. Database

```bash
# Start PostgreSQL (Docker)
docker run -d \
  --name dapta-db \
  -e POSTGRES_USER=dapta \
  -e POSTGRES_PASSWORD=dapta_password \
  -e POSTGRES_DB=dapta_db \
  -p 5432:5432 \
  postgres:16
```

### 2. Backend

```bash
cd web/api
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set SECRET_KEY and DAPTA_PATH

python main.py
# API runs on http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 3. Frontend

```bash
cd web/frontend
npm install
npm run dev
# App runs on http://localhost:5173
```

---

## API Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/signup` | — | Create account, returns token pair |
| POST | `/api/auth/login` | — | Login, returns token pair |
| POST | `/api/auth/refresh` | — | Swap refresh token for new pair |
| POST | `/api/auth/logout` | ✓ | Sign out (client discards tokens) |
| GET | `/api/auth/me` | ✓ | Get current user |
| PATCH | `/api/auth/me` | ✓ | Update clinical profile |
| GET | `/api/sessions/` | ✓ | List sessions |
| GET | `/api/sessions/:id` | ✓ | Get one session |
| DELETE | `/api/sessions/:id` | ✓ | Delete session |
| POST | `/api/assessment/text` | ✓ | Assess from text |
| POST | `/api/assessment/upload` | ✓ | Assess from .cha or audio file |
| GET | `/api/recommendations/:id` | ✓ | Get exercise recommendation |
| POST | `/api/recommendations/:id/submit-audio` | ✓ | Submit exercise, get feedback |
| GET | `/api/health` | — | Health check |

---

## Authentication Flow

1. `POST /signup` or `POST /login` → returns `{ access_token, refresh_token }`
2. Client stores both in `localStorage`
3. Every request attaches `Authorization: Bearer <access_token>`
4. On 401, client automatically calls `POST /refresh` once
5. If refresh fails → tokens cleared → redirect to `/login`

Access tokens expire in **60 minutes**. Refresh tokens expire in **30 days**.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://dapta:dapta_password@localhost:5432/dapta_db` | PostgreSQL DSN |
| `SECRET_KEY` | `CHANGE_ME_IN_PRODUCTION` | JWT signing key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `30` | Refresh token TTL |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:3000` | CORS origins |
| `DAPTA_PATH` | `../dapta` | Path to DAPTA ML library |
| `DAPTA_MODELS_PATH` | `../models` | Path to trained model checkpoints |
| `ENVIRONMENT` | `development` | `development` or `production` |

---

## Before DAPTA Models Are Trained

`DAPTAService` degrades gracefully — if model files are not found it returns
mock metric values and a mock recommendation. The entire web app is fully
usable and testable before any training has run.

Once models are trained:
1. Set `DAPTA_PATH` and `DAPTA_MODELS_PATH` in `.env`
2. Restart the API — `DAPTAService.load()` picks them up automatically
