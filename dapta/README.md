# DAPTA — Discourse-Aware Personalised Therapy Agent

A modular AI system for personalised aphasia rehabilitation therapy sequencing,
built on the AphasiaBank dataset. This is the implementation for the master's
thesis: *"Discourse-Aware Reinforcement Learning for Personalised Aphasia Therapy
Sequencing: Training and Evaluation on AphasiaBank"*

---

## System Architecture

```
DAPTA
├── Module 1: Discourse Assessment Engine (DAE)
│   ├── CHAT transcript parser
│   ├── Discourse metric extractor (CIU, MC, MLU, TTR, SynComp)
│   ├── RoBERTa fine-tuner & surprisal scorer
│   └── Patient state vector builder
│
├── Module 2: Patient Environment Simulator (PES)
│   ├── Transition model (neural MLP)
│   ├── Patient cluster builder
│   └── MDP environment (Gym-compatible)
│
└── Module 3: Personalised RL Therapy Agent (PRTA)
    ├── Double DQN with GRU history encoder + Dueling head
    ├── PPO agent (comparison)
    └── Patient-specific vs generalised policy trainer
```

---

## Project Structure

```
dapta/
├── dapta/
│   ├── dae/                    # Discourse Assessment Engine
│   │   ├── parser.py           # CHAT transcript parser
│   │   ├── metrics.py          # Discourse metric computation
│   │   ├── roberta_scorer.py   # RoBERTa fine-tuning & surprisal
│   │   └── state_builder.py    # Patient state vector builder
│   ├── pes/                    # Patient Environment Simulator
│   │   ├── transition_model.py # Neural transition model (MLP)
│   │   ├── cluster.py          # Patient clustering
│   │   └── environment.py      # Gym-compatible MDP environment
│   ├── prta/                   # Personalised RL Therapy Agent
│   │   ├── action_space.py     # 12 therapy exercise definitions
│   │   ├── ddqn_agent.py       # Double DQN + GRU + Dueling head
│   │   ├── ppo_agent.py        # PPO agent (SB3 wrapper)
│   │   └── trainer.py          # Agent training orchestrator
│   └── utils/
│       ├── config.py           # Centralised config/hyperparams
│       ├── logger.py           # Structured logging
│       ├── metrics_eval.py     # Evaluation metrics (Cohen's d, etc.)
│       └── reward.py           # Reward function
├── experiments/
│   ├── run_dae.py              # Phase 1: DAE training & validation
│   ├── run_pes.py              # Phase 2a: PES transition model training
│   ├── run_rl.py               # Phase 2b: RL agent training
│   └── run_evaluation.py       # Phase 3: Full evaluation + generalisation test
├── tests/
│   ├── test_parser.py
│   ├── test_metrics.py
│   ├── test_environment.py
│   └── test_agents.py
├── configs/
│   └── default.yaml            # All hyperparameters in one place
├── scripts/
│   └── finetune_whisper.py
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .

# 2. Phase 1 — Train DAE
python experiments/run_dae.py

# 3. Phase 2 — Train PES + RL agents
python experiments/run_pes.py
python experiments/run_rl.py

# 4. Phase 3 — Full evaluation
python experiments/run_evaluation.py
```

---

## Research Questions Mapping

| RQ | Component | Experiment |
|----|-----------|------------|
| RQ1 | DAE | run_dae.py |
| RQ2 | PRTA (DDQN vs RBDE) | run_rl.py |
| RQ3 | Generalisation test | run_evaluation.py |
| RQ4 | Patient-specific vs G-DDQN | run_evaluation.py |

---

## Citation

```
MacWhinney, B., Fromm, D., Forbes, M., & Holland, A. (2011).
AphasiaBank: Methods for studying discourse. Aphasiology, 25(11), 1286–1307.
```
