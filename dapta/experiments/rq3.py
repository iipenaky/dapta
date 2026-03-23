import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("outputs/dae/per_task_metrics.csv")
test_df = df[df['split'] == 'test']

structured_tasks   = ['cookie_theft', 'cinderella', 'sandwich']
naturalistic_tasks = ['conversation']

structured = (
    test_df[test_df['task'].isin(structured_tasks)]
    .groupby('participant_id')[['ciu_rate', 'mc_score', 'mlu_morphemes']]
    .mean()
    .add_suffix('_structured')
)

naturalistic = (
    test_df[test_df['task'].isin(naturalistic_tasks)]
    .groupby('participant_id')[['ciu_rate', 'mlu_morphemes']]
    .mean()
    .add_suffix('_naturalistic')
)

combined = structured.join(naturalistic, how='inner')
print(f"Matched participants: {len(combined)}")

# Part 1 — cross-task correlation
metrics = [
    ('ciu_rate_structured',      'ciu_rate_naturalistic',      'CIU rate'),
    ('mlu_morphemes_structured', 'mlu_morphemes_naturalistic', 'MLU'),
]

print("\nPart 1: Cross-Task Transfer Correlations (n=60)")
print("-" * 55)
for col_s, col_n, label in metrics:
    r, p = stats.spearmanr(combined[col_s], combined[col_n])
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{label:12} r = {r:.3f}, p = {p:.4f} {sig}")

# Part 2 — RL improvement vs naturalistic quality
# Load your RL evaluation results
# This file should have columns: participant_id, ciu_improvement, mc_improvement
# You get this from your run_rl.py evaluation output
try:
    rl_results = pd.read_csv("outputs/rl/dapta_test_results.csv")
    
    combined_rl = combined.join(
        rl_results.set_index('participant_id')[['ciu_improvement']],
        how='inner'
    )
    
    print(f"\nPart 2: RL Improvement vs Naturalistic Quality (n={len(combined_rl)})")
    print("-" * 55)
    for col_n, label in [
        ('ciu_rate_naturalistic', 'CIU rate'),
        ('mlu_morphemes_naturalistic', 'MLU')
    ]:
        r, p = stats.spearmanr(
            combined_rl['ciu_improvement'], 
            combined_rl[col_n]
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{label:12} r = {r:.3f}, p = {p:.4f} {sig}")

except FileNotFoundError:
    print("\nPart 2 skipped — run after run_rl.py completes")
    print("Expected file: outputs/rl/dapta_test_results.csv")

combined.to_csv("outputs/rq3_transfer_analysis.csv")
print("\nSaved to outputs/rq3_transfer_analysis.csv")