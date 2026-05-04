# DAPTA — Documentation package index (readme.txt)

## SOURCE CODE (GitHub)
  https://github.com/iipenaky/dapta

  Clone:
    git clone https://github.com/iipenaky/dapta

## WHAT IS IN THIS REPOSITORY
  Root folder contains:

    dapta/          Inner Python project — thesis ML pipeline (DAE, PES, PRTA)
                    plus experiments/, data/, outputs/, requirements.txt, setup.py

    web/            Optional FastAPI API + React (Vite) frontend

  Detailed Markdown READMEs:
    dapta/README.md    Research modules, experiment phases, quickstart
    web/README.md      API routes, env vars, SQLite default, deployment notes


SYSTEM REQUIREMENTS (SUMMARY)
-----------------------------
  • Python 3.10+ for inner dapta/ and web/api/
  • Node.js 18+ and npm for web/frontend/
  • GPU strongly recommended for Mistral fine-tuning and RL training
  • Web API default database is SQLite (no Docker required); PostgreSQL optional


INSTALL — RESEARCH PIPELINE (inner dapta/)
------------------------------------------
  Change directory to the folder that contains setup.py and requirements.txt
  (named "dapta" under the repository root):

    cd dapta

  Virtual environment (recommended):

    Windows PowerShell:
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1

    Linux / macOS:
      python -m venv .venv
      source .venv/bin/activate

  Install:

    pip install --upgrade pip
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    pip install -e .

  Phase scripts (run from the same inner "dapta" directory):

    python experiments/run_dae.py
      Optional: --skip_mistral  (faster; surprisal zeros)
      Optional: --data_dir PATH_TO_APHASIABANK_TREE

    python experiments/run_pes.py

    python experiments/run_rl.py

    python experiments/run_evaluation.py

  Other experiments (plots, sensitivity, CLAN validation, etc.) live under
  experiments/ — use each script's --help where available.


INSTALL AND RUN — WEB APPLICATION (web/)
-----------------------------------------
  Full instructions: web/README.md

  Short path:

  1) Install inner ML package first (same as above: cd dapta, pip install -e .).

  2) Backend:
       cd web/api
       python -m venv .venv
       Activate venv, then: pip install -r requirements.txt
       Copy .env.example to .env and set SECRET_KEY at minimum.
       python main.py
     Default API URL: http://localhost:8000
     OpenAPI docs (development): http://localhost:8000/docs

  3) Frontend:
       cd web/frontend
       npm install
       npm run dev
     Default dev URL: http://localhost:5173

  Important:
    • Recommendations require trained artefacts under inner dapta/outputs/
      (scaler, rl checkpoints, clusterer) — see web/README.md "Where artefacts
      are loaded".
    • DATABASE_URL defaults to SQLite; PostgreSQL is optional for production.


DEPLOYMENT (PRODUCTION — POINTERS)
-----------------------------------
  • API: run behind HTTPS with a production ASGI process manager; set
    ENVIRONMENT=production and a strong SECRET_KEY; narrow CORS ALLOWED_ORIGINS.
    Adjust database engine settings if moving off SQLite (see web/database.py).
  • Frontend: npm run build; serve static assets and configure VITE_API_URL /
    client base URL to your API origin.


OTHER FILES IN documentation/
------------------------------
  MANUAL.txt — Extended manual (workflows, troubleshooting).


DATA AND CITATION
-----------------
  MacWhinney, B., Fromm, D., Forbes, M., & Holland, A. (2011).
  AphasiaBank: Methods for studying discourse. Aphasiology, 25(11), 1286–1307.

