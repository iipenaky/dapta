# DAPTA — Discourse-Aware Personalised Therapy Agent

Modular research code for personalised aphasia rehabilitation *therapy sequencing*,
designed around transcript-level discourse assessment and reinforcement learning over
simulated therapy trajectories. The implementation supports the undergraduate thesis:

**“Discourse-Aware Reinforcement Learning for Personalised Aphasia Therapy Sequencing:
Training and Evaluation on AphasiaBank.”**

---

## System architecture

```
DAPTA
├── Module 1 — Discourse Assessment Engine (DAE)
│   ├── CHAT transcript parser (`parser.py`)
│   ├── Discourse metrics: CIU rate, MC, MLU, MATTR, syntactic complexity (`metrics.py`)
│   ├── Mistral-7B (LoRA) fine-tuner and surprisal scorer (`mistral_scorer.py`)
│   └── Patient state vector assembly (`state_builder.py`)
│
├── Module 2 — Patient Environment Simulator (PES)
│   ├── Neural transition model (MLP) (`transition_model.py`)
│   ├── Patient clustering (`cluster.py`)
│   └── Gymnasium MDP environment (`environment.py`)
│
└── Module 3 — Personalised RL Therapy Agent (PRTA)
    ├── Double DQN with GRU history encoding and dueling head (`ddqn_agent.py`)
    ├── PPO (Stable-Baselines3) as a comparative baseline (`trainer.py`)
    ├── Rule-based and random baselines (`trainer.py`)
    └── Orchestration for training and evaluation (`trainer.py`)
```

Fine-tuned **Mistral** surprisal is optional for speed: Phase 1 can be run with
`--skip_mistral` (zeros for surprisal features) while preserving the metric pipeline.

---

## Project layout

Paths are relative to this directory (the repo’s inner `dapta/` project containing
`setup.py`):

```
.
├── dapta/
│   ├── dae/                    # DAE — parsing, metrics, LM surprisal, states
│   │   ├── parser.py
│   │   ├── metrics.py
│   │   ├── mistral_scorer.py
│   │   └── state_builder.py
│   ├── pes/                    # PES — clustering, transitions, environment
│   │   ├── transition_model.py
│   │   ├── cluster.py
│   │   └── environment.py
│   ├── prta/                   # PRTA — actions, DDQN, training helpers
│   │   ├── action_space.py      # Twelve therapy exercises
│   │   ├── ddqn_agent.py
│   │   └── trainer.py           # DDQN / PPO / baselines training
│   └── utils/
│       ├── config.py            # YAML config loader when a file is supplied
│       ├── logger.py
│       └── reward.py
├── experiments/
│   ├── run_dae.py              # Phase 1 — DAE pipeline
│   ├── run_pes.py              # Phase 2a — PES / transition model
│   ├── run_rl.py               # Phase 2b — RL agents and baselines
│   ├── run_evaluation.py       # Phase 3 — held-out evaluation, RQs 3–4 style analyses
│   ├── validate_clan.py        # RQ1-style alignment with external CLAN / CHAT dirs
│   ├── plot_discourse.py
│   ├── plot_results.py
│   ├── run_sensitivity_analysis.py
│   ├── predict_action.py
│   └── get_csv.py
├── data/                       # Typical location for AphasiaBank-style archives
├── outputs/                     # Artefacts produced by experiments (generated)
├── requirements.txt
├── setup.py
└── README.md
```

A separate **FastAPI + React** application that wraps parts of this stack for a
browser workflow lives one level up under `../web/` (see `../web/README.md`).

---

## Quickstart

Run from **this** directory after creating a virtual environment.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .

# Phase 1 — DAE (add --skip_mistral for a faster run without Mistral training)
python experiments/run_dae.py

# Phase 2 — PES then RL agents
python experiments/run_pes.py
python experiments/run_rl.py

# Phase 3 — evaluation (artefacts under outputs/evaluation/, etc.)
python experiments/run_evaluation.py
```

GPU is recommended for Mistral fine-tuning and RL; pass `--device cuda` where
scripts expose it.

---

## Research question mapping

| RQ | Component | Primary experiment |
|----|-----------|--------------------|
| RQ1 | DAE versus external tooling | `run_dae.py` + `validate_clan.py` |
| RQ2 | PRTA (e.g. DDQN vs RBDE / baselines) | `run_rl.py` |
| RQ3 | Generalisation / transfer | `run_evaluation.py` |
| RQ4 | Patient-specific vs generalised policies | `run_evaluation.py` |

Exact metrics and contrasts are documented in thesis text and reflected in filenames
saved under `outputs/`.

---

## Citation — AphasiaBank

```
MacWhinney, B., Fromm, D., Forbes, M., & Holland, A. (2011).
AphasiaBank: Methods for studying discourse. Aphasiology, 25(11), 1286–1307.
```

Obtain transcript data according to AphasiaBank access policy; organise CHAT-compatible
trees under paths such as `data/aphasiabank/` (see `experiments/run_dae.py --data_dir`).
