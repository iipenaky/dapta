# DAPTA — Discourse-Aware Personalised Therapy Agent

End-to-end repository for personalised aphasia therapy sequencing research and a companion web application.

This repo contains:
- the **core ML/research pipeline** in `dapta/` (DAE + PES + PRTA modules),
- a **full-stack web app** in `web/` (FastAPI + React),
- submission/support docs in `documentation/`.

The implementation supports the undergraduate thesis:
**“Discourse-Aware Reinforcement Learning for Personalised Aphasia Therapy Sequencing: Training and Evaluation on AphasiaBank.”**

---

## Repository layout

```text
.
├── dapta/            # Core Python package + experiments (inner project)
├── web/              # FastAPI backend + React frontend
├── documentation/    # readme.txt + MANUAL.txt
└── README.MD         # This file
```

---

## 1) Core pipeline (`dapta/`)

The inner `dapta/` project contains three main modules:
- **DAE** (Discourse Assessment Engine): transcript parsing, discourse metrics, Mistral-based surprisal.
- **PES** (Patient Environment Simulator): clustering + transition model + Gymnasium environment.
- **PRTA** (Personalised RL Therapy Agent): DDQN (+ GRU) and PPO/baselines.

### Install

From repository root:

```bash
cd dapta
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

### Run main experiment phases

```bash
# Phase 1: DAE
python experiments/run_dae.py
# Optional faster run:
# python experiments/run_dae.py --skip_mistral

# Phase 2a: PES
python experiments/run_pes.py

# Phase 2b: RL
python experiments/run_rl.py

# Phase 3: Evaluation
python experiments/run_evaluation.py
```

More details: `dapta/README.md`

---

## 2) Web application (`web/`)

`web/` provides a browser workflow for auth, session management, assessment, recommendation, and feedback.

### Backend (FastAPI)

```bash
cd web/api
python -m venv .venv
# activate env
pip install -r requirements.txt
cp .env.example .env
# set SECRET_KEY at minimum
python main.py
```

Default API URL: `http://localhost:8000`  
Docs in development: `http://localhost:8000/docs`

> Note: default DB is SQLite via `DATABASE_URL=sqlite+aiosqlite:///./dapta.db`.

### Frontend (React + Vite)

```bash
cd web/frontend
npm install
npm run dev
```

Default frontend URL: `http://localhost:5173`

More details: `web/README.md`

---

## 3) Documentation (`documentation/`)

- `documentation/readme.txt` — submission-style index (repo link + install/deploy pointers)
- `documentation/MANUAL.txt` — workflow manual and troubleshooting

---

## Data and citation

This project uses AphasiaBank-style transcript data. Ensure access and usage comply with AphasiaBank policies.

Citation:

```text
MacWhinney, B., Fromm, D., Forbes, M., & Holland, A. (2011).
AphasiaBank: Methods for studying discourse. Aphasiology, 25(11), 1286–1307.
```

---

## GitHub repository

`https://github.com/iipenaky/dapta
