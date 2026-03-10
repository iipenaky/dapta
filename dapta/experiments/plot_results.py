"""
experiments/plot_results.py
---------------------------
Generates all thesis result figures from evaluation_results.json.

Produces:
  fig1_mean_improvement.png     - Table 1: bar chart, mean improvement per agent per metric
  fig2_cohens_d_heatmap.png     - Table 2: Cohen's d heatmap (DAPTA vs baselines)
  fig3_rq3_generalisation.png   - RQ3: DAPTA vs RBDE with CI bars
  fig4_rq4_personalisation.png  - RQ4: DAPTA vs G-DDQN CIU comparison
  fig5_wilcoxon_summary.png     - Statistical significance summary
  fig6_roberta_loss.png         - RoBERTa fine-tuning loss curve (bonus)

Usage:
  python experiments/plot_results.py
  python experiments/plot_results.py --eval_dir outputs/evaluation --out_dir outputs/figures
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ── Colours ──────────────────────────────────────────────────────────────────
COLOURS = {
    "DAPTA":   "#C4501A",
    "G_DDQN":  "#2563A8",
    "RBDE":    "#16854A",
    "RTS":     "#7C5FA8",
    "PPO":     "#C4951A",
}

METRIC_LABELS = {
    "CIU_rate":           "CIU Rate",
    "MC_score":           "MC Score",
    "MLU_morphemes":      "MLU-m",
    "TTR":                "TTR",
    "SynComp":            "Syn. Complexity",
    "Surprisal":          r"Surprisal ($\downarrow$)",
}

AGENTS_ORDER = ["DAPTA", "G_DDQN", "RBDE", "RTS", "PPO"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(eval_dir: Path) -> dict:
    path = eval_dir / "evaluation_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    with open(path) as f:
        return json.load(f)


def available_agents(stats: dict) -> list:
    return [a for a in AGENTS_ORDER if a in stats["per_agent_means"]]


# ── Figure 1: Mean improvement bar chart ─────────────────────────────────────

def plot_mean_improvement(stats: dict, out_dir: Path) -> None:
    agents   = available_agents(stats)
    metrics  = list(METRIC_LABELS.keys())
    n_m      = len(metrics)
    n_a      = len(agents)

    x      = np.arange(n_m)
    width  = 0.8 / n_a
    offsets = np.linspace(-(n_a - 1) / 2, (n_a - 1) / 2, n_a) * width

    fig, ax = plt.subplots(figsize=(13, 5))

    for idx, agent in enumerate(agents):
        means = [stats["per_agent_means"][agent].get(m, 0.0) for m in metrics]
        stds  = [stats["per_agent_stds"][agent].get(m, 0.0)  for m in metrics]
        bars  = ax.bar(
            x + offsets[idx], means, width * 0.9,
            color=COLOURS[agent], alpha=0.88,
            label=agent.replace("_", "-"),
            zorder=3,
        )
        ax.errorbar(
            x + offsets[idx], means, yerr=stds,
            fmt="none", color="black", linewidth=0.8,
            capsize=2, alpha=0.6, zorder=4,
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Mean improvement (normalised units)")
    ax.set_title("Figure 1 — Mean Discourse Metric Improvement by Agent\n"
                 "(positive = improvement; error bars = ±1 SD)", pad=12)
    ax.legend(framealpha=0.9, fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig1_mean_improvement.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Cohen's d heatmap ───────────────────────────────────────────────

def plot_cohens_d_heatmap(stats: dict, out_dir: Path) -> None:
    agents   = available_agents(stats)
    baselines = [a for a in agents if a != "DAPTA"]
    metrics   = list(METRIC_LABELS.keys())

    n_b = len(baselines)
    n_m = len(metrics)

    d_matrix = np.zeros((n_m, n_b))
    sig_matrix = np.zeros((n_m, n_b), dtype=bool)

    for j, bl in enumerate(baselines):
        key = f"DAPTA_vs_{bl}"
        for i, m in enumerate(metrics):
            cell = stats.get("cohens_d", {}).get(key, {}).get(m, {})
            d_matrix[i, j] = cell.get("d", 0.0)
            sig_matrix[i, j] = cell.get("clinically_meaningful", False)

    fig, ax = plt.subplots(figsize=(3 + n_b * 1.6, 1.2 + n_m * 0.7))

    vmax = max(abs(d_matrix).max(), 0.8)
    im = ax.imshow(d_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_b))
    ax.set_xticklabels([f"vs {b.replace('_','-')}" for b in baselines], fontsize=10)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)

    for i in range(n_m):
        for j in range(n_b):
            d = d_matrix[i, j]
            mark = r" $\checkmark$" if sig_matrix[i, j] else ""
            ax.text(j, i, f"{d:+.2f}{mark}", ha="center", va="center",
                    fontsize=9, fontfamily="monospace",
                    color="white" if abs(d) > vmax * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d", fontsize=9)

    ax.set_title(
        "Figure 2 — Cohen's d: DAPTA vs Baselines\n"
        "✓ = clinically meaningful (|d| ≥ 0.40, iTalkBetter benchmark)", pad=12
    )

    plt.tight_layout()
    path = out_dir / "fig2_cohens_d_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: RQ3 Generalisation ─────────────────────────────────────────────

def plot_rq3(stats: dict, out_dir: Path) -> None:
    rq3     = stats.get("rq3_generalisation", {})
    metrics = [m for m in list(METRIC_LABELS.keys())[:5] if m in rq3]

    dapta_means = [rq3[m]["dapta_mean"] for m in metrics]
    rbde_means  = [rq3[m]["rbde_mean"]  for m in metrics]
    ci_lo       = [rq3[m]["ci_95"][0]   for m in metrics]
    ci_hi       = [rq3[m]["ci_95"][1]   for m in metrics]
    sig         = [rq3[m]["null_rejected"] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - w/2, dapta_means, w, color=COLOURS["DAPTA"], label="DAPTA", alpha=0.88, zorder=3)
    ax.bar(x + w/2, rbde_means,  w, color=COLOURS["RBDE"],  label="RBDE",  alpha=0.88, zorder=3)

    # CI bars on DAPTA
    diff = np.array(dapta_means) - np.array(rbde_means)
    err_lo = diff - np.array(ci_lo)
    err_hi = np.array(ci_hi) - diff
    ax.errorbar(x - w/2, dapta_means,
                yerr=[np.abs(err_lo), np.abs(err_hi)],
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

    # Significance markers
    for i, s in enumerate(sig):
        if s:
            ymax = max(dapta_means[i], rbde_means[i]) + 0.005
            ax.text(x[i], ymax + 0.002, "* p<.05", ha="center",
                    fontsize=8, color=COLOURS["DAPTA"], fontfamily="monospace")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=15, ha="right")
    ax.set_ylabel("Mean discourse improvement")
    ax.set_title(
        "Figure 3 — RQ3: Generalisation Test (DAPTA vs RBDE)\n"
        "* = statistically significant AND clinically meaningful (d ≥ 0.40)", pad=12
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig3_rq3_generalisation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: RQ4 Personalisation ────────────────────────────────────────────

def plot_rq4(stats: dict, out_dir: Path) -> None:
    rq4 = stats.get("rq4_personalisation", {})
    if not rq4:
        print("  Skipping fig4 — no rq4 data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: mean CIU comparison with CI
    ax = axes[0]
    agents_vals = {
        "DAPTA":   rq4["dapta_ciu_mean"],
        "G-DDQN":  rq4["gddqn_ciu_mean"],
    }
    colours = [COLOURS["DAPTA"], COLOURS["G_DDQN"]]
    bars = ax.bar(agents_vals.keys(), agents_vals.values(),
                  color=colours, alpha=0.88, width=0.4, zorder=3)

    # CI on DAPTA
    lo, hi = rq4["ci_95"]
    d_mean = rq4["dapta_ciu_mean"]
    # Ensure error bar lengths are non-negative
    lower_err = max(0, d_mean - lo)
    upper_err = max(0, hi - d_mean)
    ax.errorbar(0, d_mean, yerr=[[lower_err], [upper_err]],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    ax.set_ylabel("Mean CIU rate improvement")
    ax.set_title("CIU Rate Improvement\nDAPTA (patient-specific) vs G-DDQN")
    ax.text(0.5, 0.92,
            f"d = {rq4['cohens_d_ciu']:.3f}  p = {rq4['p_value']:.4f}",
            transform=ax.transAxes, ha="center", fontsize=9,
            fontfamily="monospace",
            color=COLOURS["DAPTA"] if rq4["personalisation_beneficial"] else "gray")

    # Right: variance comparison
    ax2 = axes[1]
    vars_data = {
        "DAPTA":  rq4["dapta_variance"],
        "G-DDQN": rq4["gddqn_variance"],
    }
    ax2.bar(vars_data.keys(), vars_data.values(), color=colours, alpha=0.88, width=0.4, zorder=3)
    ax2.set_ylabel("Variance of CIU improvement")
    ax2.set_title(f"Outcome Variance\n({rq4['variance_reduction_pct']:+.1f}% change DAPTA vs G-DDQN)")

    fig.suptitle("Figure 4 — RQ4: Personalisation Test (DAPTA vs G-DDQN)", y=1.02, fontsize=13)
    plt.tight_layout()
    path = out_dir / "fig4_rq4_personalisation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 7: RL training / learning curves ───────────────────────────────────

def plot_rl_learning_curves(rl_dir: Path, out_dir: Path) -> None:
    path = rl_dir / "training_logs.json"
    if not path.exists():
        print(f"  Skipping fig7 — RL training logs not found: {path}")
        return

    with open(path) as f:
        logs = json.load(f)

    # Plot reward and loss curves. Show individual cluster curves faded and the generalised model highlighted.
    cluster_keys = sorted([k for k in logs.keys() if k.startswith("ddqn_cluster_")])
    general_key = "ddqn_generalised"

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Mean reward
    ax = axes[0]
    for k in cluster_keys:
        entry = logs[k]
        ax.plot(entry["steps"], entry["mean_reward"], color="#BBBBBB", alpha=0.4, linewidth=1)

    if general_key in logs:
        entry = logs[general_key]
        ax.plot(entry["steps"], entry["mean_reward"], color=COLOURS["G_DDQN"], linewidth=2.2, label="G-DDQN")

    ax.set_ylabel("Mean reward")
    ax.set_title("Figure 7a — DDQN training: mean reward over time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Loss
    ax = axes[1]
    for k in cluster_keys:
        entry = logs[k]
        ax.plot(entry["steps"], entry.get("loss", []), color="#BBBBBB", alpha=0.4, linewidth=1)

    if general_key in logs:
        entry = logs[general_key]
        ax.plot(entry["steps"], entry.get("loss", []), color=COLOURS["G_DDQN"], linewidth=2.2, label="G-DDQN")

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    ax.set_title("Figure 7b — DDQN training: loss over time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig7_rl_learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 8: Improvement histograms per metric ─────────────────────────────

def plot_improvement_histograms(eval_dir: Path, out_dir: Path) -> None:
    stats = load_results(eval_dir)
    agents = available_agents(stats)

    metric_keys = list(METRIC_LABELS.keys())

    improvements = {}
    for agent in agents:
        p = eval_dir / f"improvements_{agent}.npy"
        if not p.exists():
            print(f"  Skipping histograms — missing {p}")
            return
        improvements[agent] = np.load(p)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, metric in enumerate(metric_keys):
        ax = axes[i]
        for agent in agents:
            data = improvements[agent][:, i]
            ax.hist(data, bins=20, alpha=0.5, density=True,
                    label=agent.replace("_", "-"), color=COLOURS.get(agent, "#444444"))
        ax.set_title(METRIC_LABELS[metric])
        if i == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel("Improvement")
        ax.set_ylabel("Density")

    fig.suptitle("Figure 8 — Distribution of per-patient improvements by metric")
    path = out_dir / "fig8_improvement_histograms.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 9: Example patient trajectory (state/action) ─────────────────────

def plot_example_trajectory(out_dir: Path) -> None:
    path = out_dir / "fig9_example_trajectory.txt"
    with open(path, "w") as f:
        f.write("Example patient trajectory not generated.\n")
        f.write("Run `experiments/show_patient_trajectory.py` to generate it.\n")
    print(f"  Created placeholder: {path}")


# ── Figure 5: Wilcoxon significance summary ───────────────────────────────────

def plot_wilcoxon(stats: dict, out_dir: Path) -> None:
    wilcoxon = stats.get("wilcoxon", {})
    if not wilcoxon:
        print("  Skipping fig5 — no wilcoxon data.")
        return

    comparisons = list(wilcoxon.keys())
    metrics     = list(METRIC_LABELS.keys())

    n_c = len(comparisons)
    n_m = len(metrics)

    p_matrix   = np.ones((n_m, n_c))
    sig_matrix = np.zeros((n_m, n_c), dtype=bool)

    for j, comp in enumerate(comparisons):
        for i, m in enumerate(metrics):
            cell = wilcoxon[comp].get(m, {})
            p_matrix[i, j]   = cell.get("p_corrected", 1.0)
            sig_matrix[i, j] = cell.get("significant", False)

    fig, ax = plt.subplots(figsize=(3 + n_c * 2, 1.5 + n_m * 0.65))

    # Show -log10(p) for colour intensity
    logp = -np.log10(np.clip(p_matrix, 1e-4, 1.0))
    im = ax.imshow(logp, cmap="Blues", vmin=0, vmax=4, aspect="auto")

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(
        [c.replace("DAPTA_vs_", "vs ").replace("_", "-") for c in comparisons],
        fontsize=10, rotation=15, ha="right"
    )
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)

    for i in range(n_m):
        for j in range(n_c):
            p = p_matrix[i, j]
            mark = "✓" if sig_matrix[i, j] else "✗"
            colour = "white" if logp[i, j] > 2 else "black"
            ax.text(j, i, f"{mark}\np={p:.3f}", ha="center", va="center",
                    fontsize=8, fontfamily="monospace", color=colour, linespacing=1.4)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("−log₁₀(p corrected)", fontsize=9)

    ax.set_title(
        "Figure 5 — Wilcoxon Signed-Rank Tests (Bonferroni corrected)\n"
        "✓ = significant at α=0.05 after correction", pad=12
    )

    plt.tight_layout()
    path = out_dir / "fig5_wilcoxon.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: RoBERTa loss curve ──────────────────────────────────────────────

def plot_roberta_loss(roberta_dir: Path, out_dir: Path) -> None:
    state_file = roberta_dir / "trainer_state.json"
    if not state_file.exists():
        print(f"  Skipping fig6 — {state_file} not found.")
        return

    with open(state_file) as f:
        state = json.load(f)

    history = state.get("log_history", [])
    train_epochs, train_losses = [], []
    eval_epochs,  eval_losses  = [], []

    for entry in history:
        epoch = entry.get("epoch", 0)
        if "eval_loss" in entry:
            eval_epochs.append(epoch)
            eval_losses.append(entry["eval_loss"])
        elif "loss" in entry and "train_loss" not in entry:
            train_epochs.append(epoch)
            train_losses.append(entry["loss"])

    if not eval_losses:
        print("  Skipping fig6 — no loss history found in trainer_state.json.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if train_losses:
        ax.plot(train_epochs, train_losses, color="#2563A8", linewidth=2,
                label="Train loss", alpha=0.8)
        ax.scatter(train_epochs, train_losses, color="#2563A8", s=30, zorder=4)

    ax.plot(eval_epochs, eval_losses, color="#C4501A", linewidth=2.5,
            linestyle="--", label="Eval loss (val set)", alpha=0.9)
    ax.scatter(eval_epochs, eval_losses, color="#C4501A", s=50, zorder=5)

    # Best eval marker
    best_idx  = int(np.argmin(eval_losses))
    best_ep   = eval_epochs[best_idx]
    best_loss = eval_losses[best_idx]
    ax.annotate(
        f"Best: {best_loss:.3f}",
        xy=(best_ep, best_loss),
        xytext=(best_ep + 0.3, best_loss + 0.05),
        fontsize=9, fontfamily="monospace", color="#C4501A",
        arrowprops=dict(arrowstyle="->", color="#C4501A", lw=1.2),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MLM Loss")
    ax.set_title(
        "Figure 6 — RoBERTa Fine-tuning: Train vs Validation Loss\n"
        "roberta-base → AphasiaBank MLM domain adaptation", pad=10
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig6_roberta_loss.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot DAPTA thesis results")
    parser.add_argument("--eval_dir",    default="outputs/evaluation",
                        help="Directory containing evaluation_results.json")
    parser.add_argument("--rl_dir",      default="outputs/rl",
                        help="Directory containing RL training logs")
    parser.add_argument("--roberta_dir", default="outputs/dae/roberta_checkpoint",
                        help="RoBERTa checkpoint dir (contains trainer_state.json)")
    parser.add_argument("--out_dir",     default="outputs/figures",
                        help="Where to save figures")
    args = parser.parse_args()

    eval_dir    = Path(args.eval_dir)
    rl_dir      = Path(args.rl_dir)
    roberta_dir = Path(args.roberta_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from {eval_dir}/evaluation_results.json ...")
    stats = load_results(eval_dir)

    print("\nGenerating figures:")
    plot_mean_improvement(stats, out_dir)
    plot_cohens_d_heatmap(stats, out_dir)
    plot_rq3(stats, out_dir)
    plot_rq4(stats, out_dir)
    plot_rl_learning_curves(rl_dir, out_dir)
    plot_improvement_histograms(eval_dir, out_dir)
    plot_example_trajectory(out_dir)
    plot_wilcoxon(stats, out_dir)
    plot_roberta_loss(roberta_dir, out_dir)

    print(f"\nAll figures saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()