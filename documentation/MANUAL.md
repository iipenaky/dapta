# DAPTA — Manual (overview, workflows, outputs, troubleshooting)

1. Purpose
-----------
  DAPTA (Discourse-Aware Personalised Therapy Agent) assists research and,
  optionally, clinical-style workflows via a web UI: it derives discourse-
  oriented features from conversational transcripts, simulates therapy
  transitions, and learns or applies policies for sequencing exercises.

  This manual complements documentation/readme.txt and dapta/README.md.


2. Components at a glance
-------------------------
  Module  DAE — Discourse Assessment Engine
          Parses transcripts, computes metrics (e.g. CIU-related signals,
          MLU, complexity), integrates language-model surprisal when enabled,
          and builds fixed-size patient state vectors.

  Module  PES — Patient Environment Simulator
          Clusters patients, fits a neural transition model, and exposes a
          Gymnasium-style environment for rollout-based training.

  Module  PRTA — Personalised RL Therapy Agent
          Trains reinforcement-learning agents (e.g. DDQN with GRU history,
          PPO baselines) against the simulated environment.


3. Typical research workflow (command line)
-------------------------------------------
  Prerequisites: Python environment installed as in readme.txt; GPU optional
  but advised for LM and RL workloads.

  Step A — Prepare data
          Place authorised AphasiaBank (or compatible CHAT-format) corpora so
          that --data_dir points at the folder tree expected by run_dae.py
          (default: data/aphasiabank under the inner dapta project).

  Step B — Phase 1 (DAE)
          Run: python experiments/run_dae.py
          Faster dry run without fine-tuning a large LM:
            python experiments/run_dae.py --skip_mistral
          Artefacts typically land under outputs/ or paths logged at the end of
          the run (inspect console output).

  Step C — Phase 2a (PES)
          Run: python experiments/run_pes.py
          Tune --n_augment and --epochs for speed vs fidelity.

  Step D — Phase 2b (RL)
          Run: python experiments/run_rl.py
          Use smaller --total_steps for smoke tests.

  Step E — Phase 3 (evaluation)
          Run: python experiments/run_evaluation.py

  Scripts such as experiments/plot_results.py and experiments/plot_discourse.py
  consume saved metrics for thesis-style figures.


4. Web application workflow (operators)
---------------------------------------
  Preconditions:
  • PostgreSQL running with credentials matching DATABASE_URL in web/api/.env
  • Backend and frontend configured (see readme.txt Deployment section).

  Nominal sequence for an end user (after registering/logging in):

  (1) Create or resume a therapy “session”.
  (2) Provide discourse input via text upload or recording (depending on UI).
  (3) Obtain an assessed state / metrics preview.
  (4) Receive an exercise recommendation from the recommendation endpoint.
  (5) Optionally submit exercise audio/text for automated feedback.

  If trained model checkpoints are missing, the API may return placeholder
  values so UX and auth flows remain testable — replace placeholders by
  training the pipeline and setting DAPTA_MODELS_PATH (see web/README.md).


5. Logs and reproducibility
---------------------------
  • Experiment scripts emit structured logs; run_rl uses logs/run_rl.log by code
    convention — check sibling log files under logs/ after runs.
  • Fix random seeds only if exposed in scripts or config files you edit;
    document any manual seeding in your thesis or lab notebook.
  • For citation of AphasiaBank and thesis context, reuse blocks from
    dapta/README.md.


6. Troubleshooting
------------------
  Import errors for dapta
    Run pip install -e . from the inner `dapta` directory containing setup.py.

  spaCy warnings or missing model
    Run: python -m spacy download en_core_web_sm

  CUDA out of memory
    Use --device cpu, reduce batch sizes via script flags if available, or
    use --skip_mistral for Phase 1.

  HF model access errors
    Accept the model licence on Hugging Face and run huggingface-cli login.

  Web API database connection refused
    Ensure PostgreSQL is listening on the host/port in DATABASE_URL and that
    firewall rules allow the connection.

  CORS errors in browser
    Add your frontend origin to ALLOWED_ORIGINS in web/api/.env.

  Frontend cannot reach API
    Confirm API base URL in the frontend configuration matches deployment and
    that JWT tokens are refreshed per web/README.md auth flow.


7. Related files
----------------
  repository documentation/readme.txt   installation and deployment index
  dapta/README.md                       ML architecture quickstart
  web/README.md                         API routes, env vars, stack layout
