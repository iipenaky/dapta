# DAPTA Web — Full-Stack Application

Discourse-Aware Personalised Therapy Agent — web interface with authentication, session management, and the four-step therapy flow (assess → recommend → exercise → feedback).

---

## Project structure

```
web/
├── api/                        # FastAPI backend
│   ├── core/
│   │   ├── config.py           # Env / settings (pydantic-settings)
│   │   ├── security.py         # bcrypt hashing + JWT create/verify
│   │   ├── dependencies.py     # FastAPI Depends() providers
│   │   └── middleware.py       # Global error-handler middleware
│   ├── models/
│   │   ├── user.py             # User ORM (SQLAlchemy)
│   │   └── session.py          # Session ORM
│   ├── schemas/
│   │   ├── auth.py             # Auth request/response models
│   │   └── session.py          # Assessment, recommendation, feedback shapes
│   ├── services/
│   │   ├── user_service.py
│   │   ├── session_service.py
│   │   └── dapta_service.py    # DAPTA ML singleton (startup load)
│   ├── routers/
│   │   ├── auth.py             # POST /signup, /login, /refresh, /logout; GET/PATCH /me
│   │   ├── sessions.py         # GET /, GET/{id}, DELETE/{id}
│   │   ├── assessment.py       # POST /text, POST /upload
│   │   └── recommendations.py  # GET /{session_id}, POST /{session_id}/submit-audio
│   ├── utils/
│   │   └── audio_to_cha.py     # Whisper text → provisional CHAT (audio pipeline)
│   ├── database.py             # Async engine + metadata
│   ├── main.py                 # App factory, lifespan, router wiring
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/                   # React + Vite
    ├── src/
    │   ├── api/
    │   │   └── client.js       # Fetch helper, tokens, refresh on 401
    │   ├── context/
    │   │   └── AuthContext.jsx
    │   ├── hooks/
    │   │   ├── useSession.js
    │   │   └── useAudioRecorder.js
    │   ├── pages/
    │   │   ├── LoginPage.jsx
    │   │   ├── SignupPage.jsx
    │   │   ├── DashboardPage.jsx
    │   │   ├── SessionPage.jsx
    │   │   └── ProfilePage.jsx
    │   ├── components/
    │   │   ├── auth/
    │   │   ├── layout/
    │   │   ├── session/
    │   │   ├── input/
    │   │   ├── shared/
    │   │   │   └── index.js
    │   │   └── AudioRecorder.jsx
    │   ├── utils/
    │   │   ├── constants.js
    │   │   └── styles.js
    │   ├── styles/
    │   │   └── global.css
    │   ├── App.jsx
    │   └── main.jsx
    ├── index.html
    ├── vite.config.js
    └── package.json
```

---

## Setup

### 1. Database

**Default (no extra services):** the API uses **SQLite** — `DATABASE_URL` defaults to `sqlite+aiosqlite:///./dapta.db` (file created relative to the current working directory when you run the app; typical layout is alongside `main.py` under `web/api`). No Docker step required.

**Optional — PostgreSQL** (e.g. production): start a server and set `DATABASE_URL` to a `postgresql+asyncpg://…` DSN. Note: `database.py` currently passes SQLite-oriented `connect_args`; you may need to adjust the engine setup for Postgres.

Example PostgreSQL container:

```bash
docker run -d \
  --name dapta-db \
  -e POSTGRES_USER=dapta \
  -e POSTGRES_PASSWORD=dapta_password \
  -e POSTGRES_DB=dapta_db \
  -p 5432:5432 \
  postgres:16
```

### 2. Backend

The API imports the **same `dapta` Python package** as the research codebase (DAE, Whisper-assisted paths, checkpoints). Install the inner ML project first, then API extras.

From the repository root (parent of `web/` and inner `dapta/`):

```bash
cd dapta
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .

cd ../web/api
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set SECRET_KEY at minimum

python main.py
# API → http://localhost:8000
# Docs → http://localhost:8000/docs (disabled when ENVIRONMENT=production)
```

### 3. Frontend

```bash
cd web/frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## API routes

Routers are mounted with prefix `/api/...` as declared in `main.py`. Paths below omit the host.

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| POST | `/api/auth/signup` | — | Sign up → token pair (`token_type`: `bearer`) |
| POST | `/api/auth/login` | — | Login → token pair |
| POST | `/api/auth/refresh` | — | New pair from refresh token |
| POST | `/api/auth/logout` | ✓ | Stateless logout (discard tokens client-side) |
| GET | `/api/auth/me` | ✓ | Current user |
| PATCH | `/api/auth/me` | ✓ | Update clinical profile |
| GET | `/api/sessions/` | ✓ | List sessions (`limit` / `offset` query params) |
| GET | `/api/sessions/{session_id}` | ✓ | One session |
| DELETE | `/api/sessions/{session_id}` | ✓ | Delete session |
| POST | `/api/assessment/text` | ✓ | Assess pasted text |
| POST | `/api/assessment/upload` | ✓ | Upload `.cha` or audio (see backend for supported suffixes) |
| GET | `/api/recommendations/{session_id}` | ✓ | Recommendation for that session |
| POST | `/api/recommendations/{session_id}/submit-audio` | ✓ | Post-exercise audio → feedback |
| GET | `/api/health` | — | Basic health payload |

---

## Authentication flow

1. `POST /api/auth/signup` or `POST /api/auth/login` → `{ access_token, refresh_token, token_type }`.
2. Client stores tokens (e.g. `localStorage`).
3. Requests send `Authorization: Bearer <access_token>`.
4. On `401`, the bundled client attempts `POST /api/auth/refresh` with `{ refresh_token }` once (with queuing for concurrent requests).
5. If refresh fails, tokens are cleared and the SPA navigates to `/login`.

Defaults in config: **access tokens 60 minutes**, **refresh tokens 30 days** (overridable via env vars).

---

## Environment variables

Values below match `core/config.py` / `.env.example`. Override via `.env` or deployment env.

| Variable | Typical default | Description |
|----------|-----------------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./dapta.db` | Async SQLAlchemy URL |
| `SECRET_KEY` | `CHANGE_ME_IN_PRODUCTION` | JWT signing secret |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `30` | Refresh token TTL |
| `ALLOWED_ORIGINS` | `http://localhost:5173`, `http://localhost:3000` | CORS (comma-separated string or list) |
| `DAPTA_PATH` | Repository root (resolved from `core/config.py`) | Prepended to `sys.path` for `import dapta` |
| `DAPTA_MODELS_PATH` | `<repo>/outputs` (resolved path) | Whisper: subdirectory `whisper_aphasiabank`, else Hugging Face `openai/whisper-base` |
| `ENVIRONMENT` | `development` | `production` hides `/docs` and `/redoc` |

### Where artefacts are loaded

Startup logic in `services/dapta_service.py`:

- **Scaler + DDQN checkpoints + `clusterer.pkl`** resolve under **`<repo>/dapta/outputs/…`** (`dae/scaler.npz`, `rl/ddqn_cluster_*.pt`, `pes/clusterer.pkl`), **not** via `DAPTA_MODELS_PATH` alone.

So for recommendations to work you need those files under the inner ML project’s `outputs/` tree (after training or copying checkpoints).

---

## When ML components are missing

Unlike a mock fallback, **`DAPTAService` raises errors** when required pieces are missing:

- **Assessment / upload** — DAE + parser must load; otherwise metric extraction fails at request time.
- **Audio upload** — Whisper must load for transcription (`torch`, `librosa`, etc., via the ML environment).
- **Recommendation** — needs `PatientStateBuilder` scaler (`outputs/dae/`) **and** at least one loaded cluster DDQN (`outputs/rl/ddqn_cluster_*.pt`); if no agent can act, `RuntimeError` is raised. If the clusterer is absent, routing defaults to cluster `0`, so **`ddqn_cluster_0.pt`** must exist for a recommendation.

Authentication, sessions, health, and much of the UI still operate without ML artefacts; run Phase 1–3 in inner `dapta/` (see inner `README.md`) and restart the API so `load()` can pick up checkpoints.
