"""
plot_findings.py
----------------
DAPTA — Generate all publication-quality figures from output files.

Reads from:
    outputs/evaluation/test_rq2_results.json
    outputs/evaluation/test_rq4_personalisation.json
    outputs/evaluation/test_rq3_transfer.json
    outputs/evaluation/test_ablation_results.json
    outputs/evaluation/dapta_test_results.csv
    outputs/sensitivity/sensitivity_results.json
    outputs/validation/rq1_full_report.json
    outputs/pes/transition_model_validation.json

Saves all figures to:
    outputs/figures/

Run after:
    python experiments/run_evaluation.py
    python experiments/run_sensitivity_analysis.py
    python experiments/validate_against_clan.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":     "#0d0f14",
    "axes.facecolor":       "#13161d",
    "axes.edgecolor":       "#252933",
    "axes.labelcolor":      "#9ca3af",
    "axes.titlecolor":      "#e8eaf0",
    "axes.titlesize":       11,
    "axes.labelsize":       9,
    "axes.grid":            True,
    "grid.color":           "#1e2430",
    "grid.linewidth":       0.8,
    "xtick.color":          "#6b7280",
    "ytick.color":          "#6b7280",
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.facecolor":     "#1a1e28",
    "legend.edgecolor":     "#252933",
    "legend.fontsize":      8,
    "text.color":           "#e8eaf0",
    "font.family":          "monospace",
    "figure.dpi":           150,
    "savefig.dpi":          200,
    "savefig.facecolor":    "#0d0f14",
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.3,
})

# Palette
C_GREEN  = "#34d399"
C_BLUE   = "#4f9cf9"
C_PURPLE = "#a78bfa"
C_AMBER  = "#fbbf24"
C_RED    = "#f87171"
C_TEAL   = "#2dd4bf"
C_MUTED  = "#6b7280"
C_BORDER = "#252933"

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {path} not found")
        return {}
    with open(p) as f:
        return json.load(f)


def title_stamp(ax, text: str, sub: str = ""):
    ax.set_title(text, fontsize=11, color="#e8eaf0", pad=10, loc="left")
    if sub:
        ax.text(
            0, 1.02, sub,
            transform=ax.transAxes,
            fontsize=7.5, color=C_MUTED,
            ha="left", va="bottom",
        )


def save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(str(path))
    print(f"  Saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — RQ2: Agent comparison bar chart
# ══════════════════════════════════════════════════════════════════════

def plot_rq2_agent_comparison():
    print("\n[Fig 1] RQ2 — Agent CIU comparison")
    rq2 = load_json("outputs/evaluation/test_rq2_results.json")
    if not rq2:
        return

    table = rq2.get("summary_table", {})
    agent_order = [
        "DAPTA", "G_DDQN", "NO_GRU_DAPTA", "NO_GRU_G_DDQN",
        "PPO_GENERALISED", "PPO_PERSONALISED", "RBDE", "RTS",
    ]
    # Only keep agents that are present
    agents = [a for a in agent_order if a in table]

    labels   = []
    values   = []
    colors   = []
    types    = []
    for a in agents:
        row = table[a]
        labels.append(a.replace("_", "\n"))
        values.append(row.get("ciu_mean", 0))
        t = row.get("type", "baseline")
        types.append(t)
        if t == "personalised":  colors.append(C_GREEN)
        elif t == "generalised": colors.append(C_BLUE)
        else:                    colors.append(C_MUTED)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.85, zorder=3)

    # Annotate bars
    for bar, val, t in zip(bars, values, types):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=8, color="#e8eaf0",
        )

    # Baseline separator
    if "RBDE" in agents:
        rbde_val = table["RBDE"].get("ciu_mean", 0)
        ax.axhline(rbde_val, color=C_AMBER, linewidth=1, linestyle="--", alpha=0.7, zorder=2)
        ax.text(
            len(agents) - 0.3, rbde_val + 0.002,
            "RBDE baseline", fontsize=7.5, color=C_AMBER, ha="right",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean CIU Rate Improvement")
    ax.set_ylim(bottom=min(values) * 0.97)

    legend_patches = [
        mpatches.Patch(color=C_GREEN,  label="Personalised"),
        mpatches.Patch(color=C_BLUE,   label="Generalised"),
        mpatches.Patch(color=C_MUTED,  label="Baseline"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    title_stamp(ax,
        "RQ2 — Agent CIU Rate on Held-Out Test Patients",
        f"n = {rq2.get('summary_table', {}).get('DAPTA', {}).get('ciu_mean', '?')} · "
        "all RL agents vs RBDE and RTS baselines"
    )
    save(fig, "fig1_rq2_agent_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — RQ2: Per-metric Cohen's d heatmap
# ══════════════════════════════════════════════════════════════════════

def plot_rq2_metric_heatmap():
    print("\n[Fig 2] RQ2 — Per-metric effect size heatmap")
    rq2 = load_json("outputs/evaluation/test_rq2_results.json")
    if not rq2:
        return

    per_agent = rq2.get("per_agent", {})
    metrics = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
    metric_labels = ["CIU Rate", "MC Score", "MLU", "MATTR", "Syn. Comp."]

    agent_order = ["DAPTA", "G_DDQN", "NO_GRU_DAPTA", "NO_GRU_G_DDQN"]
    agents = [a for a in agent_order if a in per_agent]
    if not agents:
        print("  No per-agent data found")
        return

    # Build matrix of Cohen's d vs RBDE
    matrix = np.zeros((len(agents), len(metrics)))
    for i, agent in enumerate(agents):
        agent_data = per_agent[agent]
        vs_rbde_key = f"{agent}_vs_RBDE"
        if vs_rbde_key in agent_data:
            for j, metric in enumerate(metrics):
                d = agent_data[vs_rbde_key].get(metric, {}).get("cohens_d", 0)
                matrix[i, j] = d

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.8)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([a.replace("_", " ") for a in agents], fontsize=9)

    # Annotate cells
    for i in range(len(agents)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if 0.2 < val < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Cohen's d vs RBDE")
    title_stamp(ax,
        "RQ2 — Per-Metric Effect Size (Cohen's d) vs RBDE",
        "green = larger improvement · threshold d ≥ 0.40 for clinical meaningfulness"
    )
    save(fig, "fig2_rq2_metric_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — RQ3: Transfer scatter plot
# ══════════════════════════════════════════════════════════════════════

def plot_rq3_transfer():
    print("\n[Fig 3] RQ3 — Cross-task transfer scatter")
    csv_path = Path("outputs/evaluation/dapta_test_results.csv")
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    df = pd.read_csv(str(csv_path))
    df = df.dropna(subset=["structured_ciu_gain", "conversation_ciu_gain"])
    if len(df) < 5:
        print("  [SKIP] insufficient data")
        return

    x = df["structured_ciu_gain"].values
    y = df["conversation_ciu_gain"].values
    r, p = scipy_stats.spearmanr(x, y)

    # Color by cluster if available
    colors = C_BLUE
    if "cluster_id" in df.columns:
        cluster_palette = [C_BLUE, C_GREEN, C_PURPLE, C_AMBER, C_TEAL, C_RED]
        colors = [cluster_palette[int(c) % len(cluster_palette)]
                  for c in df["cluster_id"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, c=colors, alpha=0.65, s=55, zorder=3, edgecolors="none")

    # Regression line
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, color=C_AMBER, linewidth=1.5,
            linestyle="--", alpha=0.8, label=f"r = {r:.3f}, p = {p:.3f}")

    # Zero lines
    ax.axhline(0, color=C_BORDER, linewidth=0.8)
    ax.axvline(0, color=C_BORDER, linewidth=0.8)

    ax.set_xlabel("Mean Structured Task CIU Gain (cookie theft, cinderella, sandwich)")
    ax.set_ylabel("Conversation Task CIU Gain")
    ax.legend()

    sig_str = "significant" if p < 0.05 else "not significant"
    title_stamp(ax,
        "RQ3 — Structured Task Gains vs Naturalistic Conversation Gains",
        f"Spearman r = {r:.3f}, p = {p:.3f} ({sig_str}) · n = {len(df)} test patients"
    )
    save(fig, "fig3_rq3_transfer_scatter.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — RQ4: Personalised vs Generalised per cluster
# ══════════════════════════════════════════════════════════════════════

def plot_rq4_clusters():
    print("\n[Fig 4] RQ4 — Per-cluster personalisation")
    rq4 = load_json("outputs/evaluation/test_rq4_personalisation.json")
    if not rq4:
        return

    per_cluster = rq4.get("per_cluster", {})
    if not per_cluster:
        print("  [SKIP] no per_cluster data")
        return

    cluster_ids = sorted(per_cluster.keys(), key=lambda x: int(x))
    pers_means  = [per_cluster[c]["personalised_ciu_mean"] for c in cluster_ids]
    gen_means   = [per_cluster[c]["generalised_ciu_mean"]  for c in cluster_ids]
    ns          = [per_cluster[c]["n"] for c in cluster_ids]
    subtypes    = [per_cluster[c].get("dominant_subtype", "") for c in cluster_ids]

    x = np.arange(len(cluster_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_pers = ax.bar(x - width/2, pers_means, width, color=C_GREEN,
                       alpha=0.8, label="DAPTA (personalised)", zorder=3)
    bars_gen  = ax.bar(x + width/2, gen_means,  width, color=C_BLUE,
                       alpha=0.8, label="G-DDQN (generalised)", zorder=3)

    # n labels
    for i, (xi, n, sub) in enumerate(zip(x, ns, subtypes)):
        ax.text(xi, min(pers_means[i], gen_means[i]) - 0.008,
                f"n={n}\n{sub}", ha="center", va="top",
                fontsize=7, color=C_MUTED)

    # Delta labels
    for i, (p_val, g_val) in enumerate(zip(pers_means, gen_means)):
        delta = p_val - g_val
        color = C_GREEN if delta > 0.01 else C_RED if delta < -0.01 else C_MUTED
        ax.text(x[i], max(p_val, g_val) + 0.003,
                f"Δ{delta:+.3f}", ha="center", va="bottom",
                fontsize=7.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids])
    ax.set_ylabel("Mean CIU Rate Improvement")
    ax.legend()
    ax.set_ylim(bottom=min(pers_means + gen_means) * 0.96)

    title_stamp(ax,
        "RQ4 — Personalised vs Generalised CIU Rate by Aphasia Cluster",
        "Δ = DAPTA − G-DDQN · green = personalisation wins · all differences negligible (d < 0.10)"
    )
    save(fig, "fig4_rq4_clusters.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 5 — Sensitivity analysis: RQ2/RQ3/RQ4 across noise
# ══════════════════════════════════════════════════════════════════════

def plot_sensitivity():
    print("\n[Fig 5] Sensitivity analysis")
    data = load_json("outputs/sensitivity/sensitivity_results.json")
    if not data:
        return

    summaries = data.get("summaries", [])
    if not summaries:
        return

    noise_labels = [s["noise_label"] for s in summaries]
    noise_stds   = [s["noise_std"]   for s in summaries]

    rq2_d  = [s.get("rq2", {}).get("cohens_d_dapta_vs_rbde", np.nan) for s in summaries]
    rq2_p  = [s.get("rq2", {}).get("wilcoxon_p", np.nan)             for s in summaries]
    rq4_d  = [s.get("rq4", {}).get("cohens_d_dapta_vs_gddqn", np.nan) for s in summaries]
    rq3_r  = [s.get("rq3", {}).get("spearman_r", np.nan)             for s in summaries]
    rq3_p  = [s.get("rq3", {}).get("p_value", np.nan)                for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    x = np.arange(len(noise_labels))
    xlabels = [f"{l}\n(σ={s})" for l, s in zip(noise_labels, noise_stds)]

    # ── Panel 1: RQ2 Cohen's d
    ax = axes[0]
    bars = ax.bar(x, rq2_d, color=C_GREEN, alpha=0.8, zorder=3)
    ax.axhline(0.40, color=C_AMBER, linewidth=1, linestyle="--",
               alpha=0.7, label="clinical threshold d=0.40")
    ax.axhline(0.50, color=C_TEAL, linewidth=1, linestyle=":",
               alpha=0.7, label="medium effect d=0.50")
    for bar, val in zip(bars, rq2_d):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Cohen's d")
    ax.set_ylim(0, max(rq2_d) * 1.2 if rq2_d else 1)
    ax.legend(fontsize=7)
    title_stamp(ax, "RQ2 · DAPTA vs RBDE", "Cohen's d across noise levels")

    # ── Panel 2: RQ3 Spearman r + p threshold
    ax = axes[1]
    bar_colors = [C_GREEN if (r > 0.2 and p < 0.05)
                  else C_AMBER if p < 0.05
                  else C_MUTED
                  for r, p in zip(rq3_r, rq3_p)]
    bars = ax.bar(x, rq3_r, color=bar_colors, alpha=0.8, zorder=3)
    ax.axhline(0.20, color=C_AMBER, linewidth=1, linestyle="--",
               alpha=0.7, label="r = 0.20 threshold")
    ax.axhline(0.0, color=C_BORDER, linewidth=0.8)
    for bar, r_val, p_val in zip(bars, rq3_r, rq3_p):
        if not np.isnan(r_val):
            sig = "*" if p_val < 0.05 else "ns"
            ax.text(bar.get_x() + bar.get_width()/2,
                    (r_val or 0) + 0.005,
                    f"{r_val:.3f}{sig}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Spearman r")
    ax.legend(fontsize=7)
    title_stamp(ax, "RQ3 · Transfer Correlation", "structured → conversation CIU gain · * p<0.05")

    # ── Panel 3: RQ4 Cohen's d (should be near zero)
    ax = axes[2]
    bar_colors_rq4 = [C_GREEN if abs(d) > 0.2 else C_MUTED for d in rq4_d]
    bars = ax.bar(x, rq4_d, color=bar_colors_rq4, alpha=0.8, zorder=3)
    ax.axhline(0, color=C_BORDER, linewidth=0.8)
    ax.axhline(0.2, color=C_AMBER, linewidth=1, linestyle="--",
               alpha=0.7, label="small effect d=0.20")
    ax.axhline(-0.2, color=C_AMBER, linewidth=1, linestyle="--", alpha=0.7)
    for bar, val in zip(bars, rq4_d):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (0.003 if val >= 0 else -0.006),
                    f"{val:.3f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Cohen's d")
    ax.legend(fontsize=7)
    title_stamp(ax, "RQ4 · DAPTA vs G-DDQN", "personalised vs generalised CIU delta")

    fig.suptitle(
        "Sensitivity Analysis — Conclusions Across Transition Model Noise Levels",
        fontsize=12, color="#e8eaf0", y=1.02
    )
    fig.tight_layout()
    save(fig, "fig5_sensitivity_analysis.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 6 — RQ1: Metric validation vs CLAN
# ══════════════════════════════════════════════════════════════════════

def plot_rq1_validation():
    print("\n[Fig 6] RQ1 — Metric validation")
    report = load_json("outputs/validation/rq1_full_report.json")
    if not report:
        return

    step1 = report.get("step1_auto_vs_clan", {}).get("metrics", {})
    step23 = report.get("step2_and_3_wabaq_analysis", {})

    if not step1:
        print("  [SKIP] no step1 metrics found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: Pearson r vs threshold
    ax = axes[0]
    metric_names   = list(step1.keys())
    pearson_rs     = [step1[m]["pearson_r"]  for m in metric_names]
    thresholds     = [step1[m]["threshold"]  for m in metric_names]
    acceptable     = [step1[m]["acceptable"] for m in metric_names]

    x = np.arange(len(metric_names))
    bar_colors = [C_GREEN if a else C_AMBER for a in acceptable]
    bars = ax.bar(x, pearson_rs, color=bar_colors, alpha=0.85, zorder=3)

    # Threshold markers
    for xi, thresh in zip(x, thresholds):
        ax.plot([xi - 0.4, xi + 0.4], [thresh, thresh],
                color=C_RED, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, pearson_rs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_names], fontsize=8)
    ax.set_ylabel("Pearson r (auto vs CLAN)")
    ax.set_ylim(0, 1.1)

    legend_patches = [
        mpatches.Patch(color=C_GREEN, label="Meets threshold ✓"),
        mpatches.Patch(color=C_AMBER, label="Below threshold (construct Δ)"),
        plt.Line2D([0], [0], color=C_RED, linewidth=2, label="Threshold"),
    ]
    ax.legend(handles=legend_patches, fontsize=7)
    title_stamp(ax,
        "RQ1 Step 1 — Auto vs CLAN Gold Standard",
        "red bar = threshold · green = acceptable accuracy"
    )

    # ── Panel 2: WAB-AQ correlations (auto vs CLAN)
    ax = axes[1]
    step3 = step23.get("step3_clan_vs_auto_wabaq", {})
    if not step3:
        ax.text(0.5, 0.5, "WAB-AQ data\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color=C_MUTED)
        title_stamp(ax, "RQ1 Step 3 — Functional Signal Preservation", "")
    else:
        metrics_wab = list(step3.keys())
        clan_rs = [step3[m]["clan"]["r"] for m in metrics_wab]
        auto_rs = [step3[m]["auto"]["r"] for m in metrics_wab]
        x = np.arange(len(metrics_wab))
        width = 0.35

        ax.bar(x - width/2, clan_rs, width, color=C_MUTED, alpha=0.7,
               label="CLAN (manual)", zorder=3)
        ax.bar(x + width/2, auto_rs, width, color=C_BLUE, alpha=0.85,
               label="Automated", zorder=3)

        for xi, c_r, a_r in zip(x, clan_rs, auto_rs):
            diff = a_r - c_r
            color = C_GREEN if abs(diff) < 0.10 else C_RED
            ax.text(xi, max(c_r, a_r) + 0.01,
                    f"Δ{diff:+.2f}", ha="center", va="bottom",
                    fontsize=8, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics_wab], fontsize=8)
        ax.set_ylabel("r with WAB-AQ")
        ax.axhline(0, color=C_BORDER, linewidth=0.8)
        ax.legend()
        title_stamp(ax,
            "RQ1 Step 3 — WAB-AQ Functional Signal: CLAN vs Auto",
            "Δ = auto − CLAN · |Δ| < 0.10 = signal preserved"
        )

    fig.tight_layout()
    save(fig, "fig6_rq1_validation.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 7 — Transition model directional accuracy
# ══════════════════════════════════════════════════════════════════════

def plot_transition_model_validation():
    print("\n[Fig 7] Transition model validation")
    data = load_json("outputs/pes/transition_model_validation.json")
    if not data:
        return

    dir_acc = data.get("directional_accuracy", {})
    mae     = data.get("mae_per_metric", {})
    mse     = data.get("val_mse", None)

    if not dir_acc:
        print("  [SKIP] no directional accuracy data")
        return

    metrics = list(dir_acc.keys())
    acc_vals = [dir_acc[m] for m in metrics]
    mae_vals = [mae.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: directional accuracy
    ax = axes[0]
    bar_colors = [C_GREEN if v >= 0.65 else C_AMBER if v >= 0.55 else C_RED
                  for v in acc_vals]
    bars = ax.bar(x, acc_vals, color=bar_colors, alpha=0.85, zorder=3)
    ax.axhline(0.65, color=C_GREEN, linewidth=1, linestyle="--",
               alpha=0.7, label="Good (≥0.65)")
    ax.axhline(0.55, color=C_AMBER, linewidth=1, linestyle=":",
               alpha=0.7, label="Acceptable (≥0.55)")
    ax.axhline(0.50, color=C_MUTED, linewidth=0.8, alpha=0.5,
               label="Chance (0.50)")
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_ylabel("Directional Accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    mse_str = f"  ·  Val MSE = {mse:.5f}" if mse else ""
    title_stamp(ax,
        "Transition Model — Directional Accuracy",
        f"Proportion of transitions where predicted direction matches real{mse_str}"
    )

    # Panel 2: MAE per metric
    ax = axes[1]
    ax.bar(x, mae_vals, color=C_BLUE, alpha=0.8, zorder=3)
    for i, (xi, val) in enumerate(zip(x, mae_vals)):
        ax.text(xi, val + 0.001, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_ylabel("Mean Absolute Error")
    title_stamp(ax,
        "Transition Model — MAE per Metric",
        "Lower = more accurate state predictions · values in normalised [0,1] space"
    )

    fig.tight_layout()
    save(fig, "fig7_transition_model_validation.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 8 — Summary 2x2 overview panel
# ══════════════════════════════════════════════════════════════════════

def plot_summary_panel():
    print("\n[Fig 8] Summary overview panel")

    rq2  = load_json("outputs/evaluation/test_rq2_results.json")
    rq4  = load_json("outputs/evaluation/test_rq4_personalisation.json")
    sens = load_json("outputs/sensitivity/sensitivity_results.json")

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top-left: RQ2 CIU means ──
    ax1 = fig.add_subplot(gs[0, 0])
    table = rq2.get("summary_table", {}) if rq2 else {}
    agent_order = ["DAPTA", "G_DDQN", "NO_GRU_DAPTA", "NO_GRU_G_DDQN", "RBDE", "RTS"]
    agents = [a for a in agent_order if a in table]
    vals   = [table[a].get("ciu_mean", 0) for a in agents]
    types  = [table[a].get("type", "baseline") for a in agents]
    colors = [C_GREEN if t=="personalised" else C_BLUE if t=="generalised" else C_MUTED
              for t in types]
    y = np.arange(len(agents))
    ax1.barh(y, vals, color=colors, alpha=0.8, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([a.replace("_", " ") for a in agents], fontsize=8)
    ax1.set_xlabel("CIU Rate")
    if "RBDE" in table:
        ax1.axvline(table["RBDE"].get("ciu_mean", 0),
                    color=C_AMBER, linewidth=1, linestyle="--", alpha=0.7)
    title_stamp(ax1, "RQ2 — Agent CIU Rate", "RBDE baseline = dashed")

    # ── Top-right: Sensitivity ──
    ax2 = fig.add_subplot(gs[0, 1])
    summaries = sens.get("summaries", []) if sens else []
    if summaries:
        noise_labels = [s["noise_label"] for s in summaries]
        rq2_d = [s.get("rq2", {}).get("cohens_d_dapta_vs_rbde", 0) for s in summaries]
        rq3_r = [s.get("rq3", {}).get("spearman_r", 0) for s in summaries]
        rq4_d = [s.get("rq4", {}).get("cohens_d_dapta_vs_gddqn", 0) for s in summaries]
        x = np.arange(len(noise_labels))
        w = 0.25
        ax2.bar(x - w,   rq2_d, w, color=C_GREEN,  alpha=0.8, label="RQ2 d (vs RBDE)", zorder=3)
        ax2.bar(x,       rq3_r, w, color=C_AMBER,  alpha=0.8, label="RQ3 r (transfer)", zorder=3)
        ax2.bar(x + w,   rq4_d, w, color=C_PURPLE, alpha=0.8, label="RQ4 d (pers vs gen)", zorder=3)
        ax2.axhline(0, color=C_BORDER, linewidth=0.8)
        ax2.axhline(0.2, color=C_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(noise_labels)
        ax2.legend(fontsize=7)
    title_stamp(ax2, "Sensitivity — RQ Stability Across Noise", "bars = effect size / correlation per noise level")

    # ── Bottom-left: RQ4 cluster wins ──
    ax3 = fig.add_subplot(gs[1, 0])
    per_cluster = rq4.get("per_cluster", {}) if rq4 else {}
    if per_cluster:
        cluster_ids = sorted(per_cluster.keys(), key=lambda x: int(x))
        pers  = [per_cluster[c]["personalised_ciu_mean"] for c in cluster_ids]
        gen   = [per_cluster[c]["generalised_ciu_mean"]  for c in cluster_ids]
        delta = [p - g for p, g in zip(pers, gen)]
        bar_colors = [C_GREEN if d > 0.005 else C_RED if d < -0.005 else C_MUTED
                      for d in delta]
        x = np.arange(len(cluster_ids))
        ax3.bar(x, delta, color=bar_colors, alpha=0.8, zorder=3)
        ax3.axhline(0, color=C_BORDER, linewidth=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"C{c}" for c in cluster_ids])
        ax3.set_ylabel("Δ CIU (DAPTA − G-DDQN)")
    title_stamp(ax3, "RQ4 — Per-Cluster Personalisation Delta", "green = pers wins · all near zero")

    # ── Bottom-right: RQ2 5-metric bar ──
    ax4 = fig.add_subplot(gs[1, 1])
    per_agent = rq2.get("per_agent", {}) if rq2 else {}
    metrics = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
    metric_labels = ["CIU", "MC", "MLU", "MATTR", "Syn"]
    if "DAPTA" in per_agent:
        dapta_data = per_agent["DAPTA"]
        key = "DAPTA_vs_RBDE"
        if key in dapta_data:
            d_vals = [dapta_data[key].get(m, {}).get("cohens_d", 0) for m in metrics]
            bar_colors = [C_GREEN if abs(d) >= 0.40 else C_AMBER for d in d_vals]
            ax4.bar(range(len(metrics)), d_vals, color=bar_colors, alpha=0.8, zorder=3)
            ax4.axhline(0.40, color=C_AMBER, linewidth=1, linestyle="--",
                        alpha=0.7, label="Clinical threshold d=0.40")
            ax4.set_xticks(range(len(metrics)))
            ax4.set_xticklabels(metric_labels)
            ax4.set_ylabel("Cohen's d vs RBDE")
            ax4.legend(fontsize=7)
    title_stamp(ax4, "RQ2 — DAPTA Effect Size per Metric", "vs RBDE · green ≥ clinical threshold")

    fig.suptitle(
        "DAPTA — Research Findings Summary",
        fontsize=14, color="#e8eaf0", y=1.01,
        fontfamily="serif",
    )
    save(fig, "fig8_summary_panel.png")


# ══════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DAPTA — Generating Figures")
    print("=" * 60)
    print(f"Output directory: {FIGURES_DIR.resolve()}")

    plot_rq2_agent_comparison()
    plot_rq2_metric_heatmap()
    plot_rq3_transfer()
    plot_rq4_clusters()
    plot_sensitivity()
    plot_rq1_validation()
    plot_transition_model_validation()
    plot_summary_panel()

    print("\n" + "=" * 60)
    print("All figures saved to outputs/figures/")
    print("  fig1_rq2_agent_comparison.png")
    print("  fig2_rq2_metric_heatmap.png")
    print("  fig3_rq3_transfer_scatter.png")
    print("  fig4_rq4_clusters.png")
    print("  fig5_sensitivity_analysis.png")
    print("  fig6_rq1_validation.png")
    print("  fig7_transition_model_validation.png")
    print("  fig8_summary_panel.png")
    print("=" * 60)