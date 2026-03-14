# Aphasia RL — Complete Pipeline
### Undergraduate Thesis: AI-Assisted Aphasia Rehabilitation

---

## Correct Pipeline Logic

```
Raw .cha files
     │
     ├──→ CLAN (ground truth)         ← validation only, NOT used for RL
     │
     ├──→ Python NLP (computes metrics independently)
     │         └──→ compare vs CLAN  →  RQ1: validates Python pipeline
     │
     ├──→ LLM scoring (scores transcripts 1-5)
     │         └──→ compare vs CLAN  →  RQ1: validates LLM scorer
     │
     └── If Python validated → Python metrics = RL reward signal
                   │
                   ↓
         RL environment (simulated patients)
                   │
           ┌───────┴────────┐
      General PPO      General DQN       ← RQ2: both vs baselines
           │
     Patient-specific PPO               ← RQ4: vs general model
           │
     Held-out task evaluation           ← RQ3: functional transfer
```

---

## File Structure

```
aphasia_rl/
├── config.yaml                  ← ALL parameters (edit this, never the code)
├── requirements.txt
├── aphasia_env.py               ← Gymnasium environment
│
├── 01_parse_cha.py              ← Parse .cha → transcripts CSV
├── 02_clan_metrics.py           ← Parse CLAN outputs → ground truth CSV
├── 03_compute_metrics.py        ← Compute metrics from raw text (Python NLP)
├── 04_llm_validation.py         ← Validate Python vs CLAN, LLM vs CLAN
├── 05_build_env.py              ← Build RL environment + synthetic patients
├── 06_train_rl.py               ← Train general + patient-specific models
├── 07_evaluate.py               ← Baselines, stats, subgroup analysis
│
├── data/
│   ├── raw/                     ← PUT .cha FILES HERE
│   ├── clan_outputs/            ← PUT CLAN .cex / .eval.xls FILES HERE
│   └── processed/               ← auto-generated
├── models/
│   └── patient_specific/        ← per-patient fine-tuned models
├── results/
└── logs/
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Run Order

```bash
python 01_parse_cha.py          # Phase 1: parse .cha files
python 02_clan_metrics.py       # Phase 2: parse CLAN outputs (ground truth)
python 03_compute_metrics.py    # Phase 3: compute metrics from text
python 04_llm_validation.py     # Phase 4: validate (Python vs CLAN, LLM vs CLAN)
python 05_build_env.py          # Phase 5: build RL environment
python 06_train_rl.py           # Phase 6: train general + patient-specific models
python 07_evaluate.py           # Phase 7: evaluate, stats, plots
```

---

## What Each RQ Gets From This Code

| RQ | Script | Key output |
|---|---|---|
| RQ1 | 04 | `validation_python_vs_clan.csv` — Pearson r, Bland-Altman |
| RQ1 | 04 | `validation_llm_vs_clan.csv` — Kappa, Pearson r |
| RQ2 | 07 | `statistical_tests.csv` — Wilcoxon W, p, Cohen's d |
| RQ3 | 07 | `reward_vs_wab_aq.png` — functional transfer proxy |
| RQ4 | 06 | `general_vs_patient_specific.png` — t-test per patient |

---

## Troubleshooting

**No .cha files found** → copy files into `data/raw/`

**LLM out of memory** → change `model_name` in config.yaml to:
`"google/flan-t5-large"` (runs on CPU, much smaller)

**Optuna too slow** → set `n_trials: 5` and `total_timesteps: 20000` for a quick test

**CLAN files missing** → Phase 3 and onwards still work without CLAN.
Phase 4 will skip the validation comparisons and warn you.