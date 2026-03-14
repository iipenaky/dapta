"""
validate.py
===========
Compares our automatically-computed metrics to CLAN ground truth.
This directly answers RQ1:
  "Can discourse-level metrics be automatically computed with enough
   accuracy to serve as a therapy reward signal?"

Validation approach:
  - Pearson r >= 0.70 is the acceptance threshold
  - MAE shows the absolute error magnitude
  - Bias shows whether we systematically over- or under-count
  - Bland-Altman style interpretation in the printed report

Outputs:
  results/validation/rq1_validation_results.csv  — numbers for the thesis
  results/validation/rq1_validation_scatter.png  — figures for the thesis
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Metric pairs: (our column, CLAN column, display name)
# These are the columns that exist in both our CSV and the CLAN CSV
METRIC_PAIRS = [
    ("mlu_words",      "mlu_words_clan",   "MLU in words"),
    ("mlu_morphemes",  "mlu_morphemes",    "MLU in morphemes"),
    ("ttr",            "ttr_clan",         "Type-Token Ratio"),
    ("total_tokens",   "total_tokens_clan","Total tokens"),
    ("ndw",            "total_types_clan", "NDW (total types)"),
    ("n_utterances",   "n_utterances_mlu", "N utterances"),
]

PEARSON_THRESHOLD = 0.70


def run_validation(
    our_csv:  str,
    clan_csv: str,
    out_dir:  str,
) -> pd.DataFrame:
    """
    Load both CSVs, merge on participant_id, compute validation statistics,
    print the RQ1 results table, and save outputs.

    Parameters
    ----------
    our_csv  : path to our computed metrics CSV
    clan_csv : path to CLAN ground truth CSV (from 02_clan_metrics.py)
    out_dir  : directory to save plots and results CSV

    Returns
    -------
    pd.DataFrame with one row per metric and columns:
        metric, n, pearson_r, p_value, mae, bias, validated
    """
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_ours = pd.read_csv(our_csv)
    df_clan = pd.read_csv(clan_csv)

    print(f"\nOur metrics:  {len(df_ours)} participants")
    print(f"CLAN metrics: {len(df_clan)} participants")

    df = df_ours.merge(df_clan, on="participant_id", how="inner",
                       suffixes=("_ours", "_clan"))
    print(f"Matched:      {len(df)} participants\n")

    if len(df) == 0:
        print("ERROR: No matching participant IDs.")
        print("Make sure your .cha filenames match the participant_id values "
              "in your CLAN CSV.")
        return pd.DataFrame()

    results = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    print("=" * 80)
    print(f"{'Metric':<22} {'N':>4} {'Pearson r':>10} {'p-value':>10} "
          f"{'MAE':>9} {'Bias':>9} {'r>=0.70?':>10}")
    print("=" * 80)

    for i, (our_col, clan_col, label) in enumerate(METRIC_PAIRS):
        # Handle the column name collision from merge suffixes
        our_actual  = our_col  if our_col  in df.columns else our_col  + "_ours"
        clan_actual = clan_col if clan_col in df.columns else clan_col + "_clan"

        if our_actual not in df.columns or clan_actual not in df.columns:
            print(f"{label:<22}  columns not found — skipping")
            continue

        paired = df[[our_actual, clan_actual]].dropna()
        n      = len(paired)

        if n < 3:
            print(f"{label:<22}  only {n} paired rows — skipping")
            continue

        x = paired[clan_actual].values   # CLAN = ground truth (x-axis)
        y = paired[our_actual].values    # ours  = computed    (y-axis)

        r, p  = stats.pearsonr(x, y)
        mae   = float(np.mean(np.abs(y - x)))
        bias  = float(np.mean(y - x))
        passed = r >= PEARSON_THRESHOLD

        results.append({
            "metric":    label,
            "n":         n,
            "pearson_r": round(float(r), 4),
            "p_value":   round(float(p), 4),
            "mae":       round(mae,       4),
            "bias":      round(bias,      4),
            "validated": passed,
        })

        status = "✓ YES" if passed else "✗ NO"
        print(f"{label:<22} {n:>4} {r:>10.4f} {p:>10.4f} "
              f"{mae:>9.3f} {bias:>+9.3f} {status:>10}")

        # Scatter plot
        ax   = axes[i]
        lims = [
            min(x.min(), y.min()) * 0.93,
            max(x.max(), y.max()) * 1.07,
        ]
        ax.scatter(x, y, alpha=0.75, s=65, color="steelblue",
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.plot(lims, lims, "k--", lw=1.2, label="Perfect agreement", zorder=2)
        m_reg, b_reg = np.polyfit(x, y, 1)
        xr = np.linspace(lims[0], lims[1], 100)
        ax.plot(xr, m_reg * xr + b_reg, "r-", lw=1.8,
                label=f"Our fit  r={r:.3f}", zorder=4)
        ax.set_xlabel(f"CLAN  {label}", fontsize=9)
        ax.set_ylabel(f"Ours  {label}", fontsize=9)
        ax.set_title(f"{label}\nr={r:.3f}  MAE={mae:.3f}  bias={bias:+.3f}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    print("=" * 80)

    df_results = pd.DataFrame(results)

    # Save CSV
    val_csv = out_dir / "rq1_validation_results.csv"
    df_results.to_csv(val_csv, index=False)

    # Save plot
    plt.suptitle(
        f"RQ1 Validation: Our Metrics vs CLAN Ground Truth"
        f"  (N={len(df)} participants)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plot_path = out_dir / "rq1_validation_scatter.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    n_pass = int(df_results["validated"].sum()) if len(df_results) > 0 else 0
    print(f"\nResult: {n_pass}/{len(df_results)} metrics validated at r >= 0.70\n")

    print("Metrics VALIDATED → enter reward function and state vector:")
    for _, row in df_results[df_results["validated"]].iterrows():
        print(f"  ✓  {row['metric']:<28}  r={row['pearson_r']:.4f}  "
              f"MAE={row['mae']:.3f}  bias={row['bias']:+.3f}")

    if (~df_results["validated"]).any():
        print("\nMetrics NOT validated → noted as limitation in thesis:")
        for _, row in df_results[~df_results["validated"]].iterrows():
            print(f"  ✗  {row['metric']:<28}  r={row['pearson_r']:.4f}")

    print(f"\nSaved → {val_csv}")
    print(f"Saved → {plot_path}")

    return df_results
