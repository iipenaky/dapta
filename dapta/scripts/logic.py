import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run_validation(data_path="outputs/pes/inferred_actions.csv"):
    # 1. Load Data
    path = Path(data_path)
    if not path.exists():
        print(f"Error: {data_path} not found. Ensure your PES script has run.")
        return
    
    df = pd.read_csv(path)

    # 2. Identify the Primary Deficit for each patient
    # We look for the lowest score among your 6 normalized discourse metrics
    metrics = ['ciu_rate', 'mc_score', 'mlu_m', 'ttr', 'syn_comp', 'surprisal']
    
    # Filter only metrics that exist in your CSV
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("Error: No discourse metrics found in CSV to identify deficits.")
        return

    # Create the 'Primary_Deficit' column by finding the min value across metrics
    # For surprisal, we look for the MAX value (since high surprisal = deficit)
    # But for simplicity in this validation, let's stick to the 5 direct metrics:
    direct_metrics = ['ciu_rate', 'mc_score', 'mlu_m', 'ttr', 'syn_comp']
    df['Primary_Deficit'] = df[direct_metrics].idxmin(axis=1)

    # 3. Create Cross-tabulation
    # This shows: "For patients with deficit X, how often did we infer action Y?"
    ctab = pd.crosstab(df['Primary_Deficit'], df['inferred_action'], normalize='index')

    # 4. Plot Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(ctab, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Clinical Face Validity: Metric Deficit vs. Inferred Action")
    plt.xlabel("Inferred Therapy Action (Latent Label)")
    plt.ylabel("Baseline Patient Deficit (Lowest Metric)")
    
    # 5. Save
    output_path = Path("outputs/plots/face_validity_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Brutally honest validation saved to: {output_path}")

if __name__ == "__main__":
    run_validation()