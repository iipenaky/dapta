import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

COLOURS = {
    "DAPTA":  "#C4501A",
    "G_DDQN": "#2563A8",
    "RBDE":   "#16854A",
    "RTS":    "#7C5FA8",
    "PPO":    "#C4951A",
}

METRIC_LABELS = {
    "ciu_rate":             "CIU Rate",
    "mc_score":             "MC Score",
    "mlu_morphemes":        "MLU-m",
    "mattr":                "MATTR",
    "syntactic_complexity": "Syn. Complexity",
}

METRICS      = list(METRIC_LABELS.keys())
AGENTS_ORDER = ["DAPTA", "G_DDQN", "RBDE", "RTS", "PPO"]


def load_results(eval_dir: Path) -> dict:
    path = eval_dir / "evaluation_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_subtype(eval_dir: Path) -> dict:
    path = eval_dir / "rq4_subtype_analysis.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def available_agents(rq2: dict) -> list:
    agents = set()
    for comp_data in rq2.values():
        if isinstance(comp_data, dict):
            for metric_data in comp_data.values():
                if isinstance(metric_data, dict) and "dapta_mean" in metric_data:
                    agents.add("DAPTA")
                if isinstance(metric_data, dict) and "baseline_mean" in metric_data:
                    pass

    comparisons = [k for k in rq2.keys() if k.startswith("DAPTA_vs_")]
    agents = ["DAPTA"] + [c.replace("DAPTA_vs_", "") for c in comparisons]
    return [a for a in AGENTS_ORDER if a in agents]

def plot_mean_improvement(results: dict, out_dir: Path) -> None:
    rq2 = results["rq2"]

    agent_means = {}
    agent_means["DAPTA"] = {
        m: rq2["DAPTA_vs_RBDE"][m]["dapta_mean"]
        for m in METRICS if m in rq2["DAPTA_vs_RBDE"]
    }

    for comp_key, agent_key in [
        ("DAPTA_vs_RBDE", "RBDE"),
        ("DAPTA_vs_RTS",  "RTS"),
        ("DAPTA_vs_G_DDQN", "G_DDQN"),
        ("DAPTA_vs_PPO",  "PPO"),
    ]:
        if comp_key in rq2:
            agent_means[agent_key] = {
                m: rq2[comp_key][m]["baseline_mean"]
                for m in METRICS if m in rq2[comp_key]
            }

    # Also get G_DDQN from rq4
    rq4 = results["rq4"]
    if "pooled_dapta_vs_gddqn" in rq4:
        agent_means["G_DDQN"] = {
            m: rq4["pooled_dapta_vs_gddqn"][m]["gddqn_mean"]
            for m in METRICS if m in rq4["pooled_dapta_vs_gddqn"]
        }

    agents = [a for a in AGENTS_ORDER if a in agent_means and agent_means[a]]

    x      = np.arange(len(METRICS))
    n_a    = len(agents)
    width  = 0.8 / n_a
    offsets = np.linspace(-(n_a - 1) / 2, (n_a - 1) / 2, n_a) * width

    fig, ax = plt.subplots(figsize=(13, 5))

    for idx, agent in enumerate(agents):
        means = [agent_means[agent].get(m, 0.0) for m in METRICS]
        ax.bar(
            x + offsets[idx], means, width * 0.9,
            color=COLOURS[agent], alpha=0.88,
            label=agent.replace("_", "-"),
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=20, ha="right")
    ax.set_ylabel("Mean improvement (normalised units)")
    ax.set_title(
        "Figure 1 — Mean Discourse Metric Improvement by Agent\n"
        "(positive = improvement over episode)",
        pad=12,
    )
    ax.legend(framealpha=0.9, fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig1_mean_improvement.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Cohen's d heatmap
# ---------------------------------------------------------------------------

def plot_cohens_d_heatmap(results: dict, out_dir: Path) -> None:
    rq2 = results["rq2"]

    comparisons = [k for k in rq2.keys() if k.startswith("DAPTA_vs_")]
    baselines   = [c.replace("DAPTA_vs_", "") for c in comparisons]

    n_m = len(METRICS)
    n_b = len(baselines)

    d_matrix   = np.zeros((n_m, n_b))
    sig_matrix = np.zeros((n_m, n_b), dtype=bool)

    for j, (comp, bl) in enumerate(zip(comparisons, baselines)):
        for i, m in enumerate(METRICS):
            if m in rq2[comp]:
                d_matrix[i, j]   = rq2[comp][m].get("cohens_d", 0.0)
                sig_matrix[i, j] = rq2[comp][m].get("clinically_meaningful", False)

    fig, ax = plt.subplots(figsize=(3 + n_b * 1.8, 1.5 + n_m * 0.8))

    vmax = max(abs(d_matrix).max(), 0.8)
    im   = ax.imshow(d_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_b))
    ax.set_xticklabels(
        [f"vs {b.replace('_', '-')}" for b in baselines],
        fontsize=10,
    )
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)

    for i in range(n_m):
        for j in range(n_b):
            d    = d_matrix[i, j]
            mark = " *" if sig_matrix[i, j] else ""
            ax.text(
                j, i, f"{d:+.2f}{mark}",
                ha="center", va="center",
                fontsize=9, fontfamily="monospace",
                color="white" if abs(d) > vmax * 0.6 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d", fontsize=9)
    ax.set_title(
        "Figure 2 — Cohen's d: DAPTA vs Baselines\n"
        "* = clinically meaningful (|d| >= 0.40)",
        pad=12,
    )

    plt.tight_layout()
    path = out_dir / "fig2_cohens_d_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rq2(results: dict, out_dir: Path) -> None:
    rq2 = results["rq2"]

    if "DAPTA_vs_RBDE" not in rq2:
        print("  Skipping fig3 — DAPTA_vs_RBDE not in rq2.")
        return

    metrics     = [m for m in METRICS if m in rq2["DAPTA_vs_RBDE"]]
    dapta_means = [rq2["DAPTA_vs_RBDE"][m]["dapta_mean"]    for m in metrics]
    rbde_means  = [rq2["DAPTA_vs_RBDE"][m]["baseline_mean"] for m in metrics]
    ci_lo       = [rq2["DAPTA_vs_RBDE"][m]["ci_95"][0]      for m in metrics]
    ci_hi       = [rq2["DAPTA_vs_RBDE"][m]["ci_95"][1]      for m in metrics]
    sig         = [rq2["DAPTA_vs_RBDE"][m]["significant"]   for m in metrics]
    ds          = [rq2["DAPTA_vs_RBDE"][m]["cohens_d"]      for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - w/2, dapta_means, w, color=COLOURS["DAPTA"], label="DAPTA", alpha=0.88, zorder=3)
    ax.bar(x + w/2, rbde_means,  w, color=COLOURS["RBDE"],  label="RBDE",  alpha=0.88, zorder=3)

    # CI bars
    err_lo = np.array(dapta_means) - np.array(ci_lo)
    err_hi = np.array(ci_hi) - np.array(dapta_means)
    ax.errorbar(
        x - w/2, dapta_means,
        yerr=[np.abs(err_lo), np.abs(err_hi)],
        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4,
    )

    # Significance + effect size markers
    for i, (s, d) in enumerate(zip(sig, ds)):
        ymax = max(dapta_means[i], rbde_means[i]) + 0.01
        if s:
            ax.text(x[i], ymax + 0.002, f"* d={d:+.2f}",
                    ha="center", fontsize=8, color=COLOURS["DAPTA"],
                    fontfamily="monospace")
        else:
            ax.text(x[i], ymax + 0.002, f"d={d:+.2f}",
                    ha="center", fontsize=7, color="gray",
                    fontfamily="monospace")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=15, ha="right")
    ax.set_ylabel("Mean discourse improvement")
    ax.set_title(
        "Figure 3 — RQ2: DAPTA vs RBDE (Rule-Based Baseline)\n"
        "* = statistically significant after Bonferroni correction",
        pad=12,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig3_rq2_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rq3(results: dict, out_dir: Path) -> None:
    rq3 = results.get("rq3", {})
    si  = rq3.get("surprisal_improvement", {})

    if not si:
        print("  Skipping fig4 — no RQ3 surprisal data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: surprisal improvement comparison
    ax = axes[0]
    agents = ["DAPTA", "RBDE"]
    vals   = [si.get("dapta_mean_surp_improvement", 0), si.get("rbde_mean_surp_improvement", 0)]
    cols   = [COLOURS["DAPTA"], COLOURS["RBDE"]]
    ax.bar(agents, vals, color=cols, alpha=0.88, width=0.4, zorder=3)

    ci_lo, ci_hi = si.get("ci_95", [0, 0])
    d_val  = si.get("dapta_mean_surp_improvement", 0)
    err_lo = max(d_val - ci_lo, 0)
    err_hi = max(ci_hi - d_val, 0)
    ax.errorbar(
        0, d_val,
        yerr=[[err_lo], [err_hi]],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean surprisal improvement")
    ax.set_title(
        f"Surprisal Improvement\n"
        f"d = {si.get('cohens_d', 0):.3f}  "
        f"p = {si.get('wilcoxon_p', 1):.4f}"
        f"{'  *' if si.get('significant', False) else ''}",
    )

    # Right: CIU-surprisal correlation
    ax2 = axes[1]
    corr = rq3.get("discourse_to_surprisal_correlation", {})

    if corr and "pearson_r" in corr:
        r   = corr["pearson_r"]
        p   = corr["p_value"]
        n   = corr.get("n", "?")
        sig = corr.get("significant", False)

        # Draw a conceptual scatter proxy
        theta  = np.linspace(0, np.pi, 100)
        x_line = np.cos(theta) * 0.4 + 0.5
        y_line = np.sin(theta) * r * 0.4 + 0.5
        ax2.plot([0.1, 0.9], [0.1 + r * 0.4, 0.9 - r * 0.4 + r * 0.8],
                 color=COLOURS["DAPTA"], linewidth=2.5, alpha=0.8)

        ax2.text(0.5, 0.5,
                 f"r = {r:+.3f}\np = {p:.4f}\nn = {n}\n"
                 f"{'Significant *' if sig else 'Not significant'}",
                 ha="center", va="center", fontsize=12,
                 transform=ax2.transAxes,
                 fontfamily="monospace",
                 color=COLOURS["DAPTA"] if sig else "gray")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("CIU Rate Improvement")
        ax2.set_ylabel("Surprisal Improvement")
        ax2.set_title(f"CIU vs Surprisal Correlation\n(transfer to naturalistic speech)")
    else:
        ax2.text(0.5, 0.5, "Insufficient data\nfor correlation",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, color="gray")
        ax2.set_title("CIU vs Surprisal Correlation")

    fig.suptitle("Figure 4 — RQ3: Transfer to Naturalistic Speech (Surprisal)", fontsize=13)
    plt.tight_layout()
    path = out_dir / "fig4_rq3_transfer.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: RQ4 personalisation
# ---------------------------------------------------------------------------

def plot_rq4(results: dict, out_dir: Path) -> None:
    rq4 = results.get("rq4", {})
    pooled = rq4.get("pooled_dapta_vs_gddqn", {})

    if not pooled:
        print("  Skipping fig5 — no RQ4 pooled data.")
        return

    # Get CIU metrics
    ciu = pooled.get("ciu_rate", {})
    dapta_mean = ciu.get("dapta_mean", 0)
    gddqn_mean = ciu.get("gddqn_mean", 0)
    d_val      = ciu.get("cohens_d", 0)
    p_val      = ciu.get("p_raw", 1)
    ci_lo, ci_hi = ciu.get("ci_95", [0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: CIU mean comparison
    ax = axes[0]
    ax.bar(
        ["DAPTA", "G-DDQN"],
        [dapta_mean, gddqn_mean],
        color=[COLOURS["DAPTA"], COLOURS["G_DDQN"]],
        alpha=0.88, width=0.4, zorder=3,
    )
    err_lo = max(dapta_mean - ci_lo, 0)
    err_hi = max(ci_hi - dapta_mean, 0)
    ax.errorbar(
        0, dapta_mean,
        yerr=[[err_lo], [err_hi]],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4,
    )
    ax.set_ylabel("Mean CIU rate improvement")
    ax.set_title(
        f"CIU Rate: DAPTA vs G-DDQN\n"
        f"d = {d_val:.3f}  p = {p_val:.4f}"
        f"{'  *' if p_val < 0.05 else '  (n.s.)'}",
    )

    # Right: per-cluster breakdown
    ax2 = axes[1]
    per_cluster = rq4.get("per_cluster", {})
    if per_cluster:
        cluster_ids   = sorted(per_cluster.keys(), key=lambda x: int(x))
        cluster_labels_plot = [
            f"C{c}\n({per_cluster[c]['dominant_subtype'][:6]})"
            for c in cluster_ids
        ]
        dapta_ciu = [per_cluster[c]["dapta_ciu_mean"] for c in cluster_ids]
        gddqn_ciu = [per_cluster[c]["gddqn_ciu_mean"] for c in cluster_ids]
        ds        = [per_cluster[c]["cohens_d_ciu"]   for c in cluster_ids]

        x = np.arange(len(cluster_ids))
        w = 0.35
        ax2.bar(x - w/2, dapta_ciu, w, color=COLOURS["DAPTA"],  label="DAPTA",  alpha=0.88, zorder=3)
        ax2.bar(x + w/2, gddqn_ciu, w, color=COLOURS["G_DDQN"], label="G-DDQN", alpha=0.88, zorder=3)

        for i, d in enumerate(ds):
            ymax = max(dapta_ciu[i], gddqn_ciu[i]) + 0.005
            ax2.text(x[i], ymax, f"d={d:+.2f}",
                     ha="center", fontsize=8, fontfamily="monospace",
                     color=COLOURS["DAPTA"] if d > 0 else COLOURS["G_DDQN"])

        ax2.set_xticks(x)
        ax2.set_xticklabels(cluster_labels_plot, fontsize=9)
        ax2.set_ylabel("Mean CIU improvement")
        ax2.set_title("Per-cluster CIU: DAPTA vs G-DDQN")
        ax2.legend(fontsize=9)

    fig.suptitle("Figure 5 — RQ4: Patient-Specific vs Generalised RL", fontsize=13)
    plt.tight_layout()
    path = out_dir / "fig5_rq4_personalisation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: Subtype moderation
# ---------------------------------------------------------------------------

def plot_subtype(subtype_data: dict, out_dir: Path) -> None:
    per_subtype = subtype_data.get("per_subtype", {})
    if not per_subtype:
        print("  Skipping fig6 — no subtype data.")
        return

    ranking = subtype_data.get("subtype_ranking", [])
    if not ranking:
        ranking = [
            {"subtype": s, "mean_benefit": per_subtype[s]["mean_benefit"],
             "cohens_d": per_subtype[s]["cohens_d"]}
            for s in per_subtype
        ]
        ranking = sorted(ranking, key=lambda x: x["mean_benefit"], reverse=True)

    subtypes  = [r["subtype"] for r in ranking]
    benefits  = [r["mean_benefit"] for r in ranking]
    ds        = [r["cohens_d"] for r in ranking]
    colours   = [COLOURS["DAPTA"] if b >= 0 else COLOURS["G_DDQN"] for b in benefits]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean benefit bar chart
    ax = axes[0]
    bars = ax.barh(subtypes, benefits, color=colours, alpha=0.88, zorder=3)
    ax.axvline(0, color="black", linewidth=1.2, zorder=2)
    ax.set_xlabel("Mean CIU personalisation benefit\n(DAPTA minus G-DDQN)")
    ax.set_title("Personalisation Benefit by Aphasia Subtype")

    for bar, b in zip(bars, benefits):
        ax.text(
            b + (0.001 if b >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{b:+.4f}",
            va="center",
            ha="left" if b >= 0 else "right",
            fontsize=8, fontfamily="monospace",
        )

    # Right: Cohen's d bar chart
    ax2 = axes[1]
    d_colours = [COLOURS["DAPTA"] if d >= 0 else COLOURS["G_DDQN"] for d in ds]
    ax2.barh(subtypes, ds, color=d_colours, alpha=0.88, zorder=3)
    ax2.axvline(0,    color="black",  linewidth=1.2, zorder=2)
    ax2.axvline(0.40, color="orange", linewidth=1.0, linestyle="--",
                alpha=0.8, label="Clinical threshold (d=0.40)")
    ax2.axvline(-0.40, color="orange", linewidth=1.0, linestyle="--", alpha=0.8)
    ax2.set_xlabel("Cohen's d")
    ax2.set_title("Effect Size by Subtype")
    ax2.legend(fontsize=8)

    ns = {s: per_subtype[s]["n"] for s in subtypes if s in per_subtype}
    for i, (s, d) in enumerate(zip(subtypes, ds)):
        n = ns.get(s, "?")
        ax2.text(
            d + (0.01 if d >= 0 else -0.01),
            i,
            f"n={n}",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=8, color="gray",
        )

    fig.suptitle(
        "Figure 6 — RQ4 Supplementary: Aphasia Subtype Moderates Personalisation Benefit",
        fontsize=12,
    )
    plt.tight_layout()
    path = out_dir / "fig6_subtype_benefit.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_ppo_comparison(results: dict, ppo_disc: np.ndarray, out_dir: Path) -> None:
    """Figure 7 — PPO vs DAPTA and RBDE comparison."""
    if ppo_disc is None:
        print("  Skipping fig7 — improvements_PPO.npy not found.")
        return

    rq2 = results["rq2"]

    dapta_means = [rq2["DAPTA_vs_RBDE"][m]["dapta_mean"]    for m in METRICS if m in rq2["DAPTA_vs_RBDE"]]
    rbde_means  = [rq2["DAPTA_vs_RBDE"][m]["baseline_mean"] for m in METRICS if m in rq2["DAPTA_vs_RBDE"]]
    ppo_means   = [float(np.mean(ppo_disc[:, i])) for i in range(len(METRICS))]

    x = np.arange(len(METRICS))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w,   dapta_means, w, color=COLOURS["DAPTA"],  label="DAPTA",  alpha=0.88, zorder=3)
    ax.bar(x,       ppo_means,   w, color=COLOURS["PPO"],    label="PPO",    alpha=0.88, zorder=3)
    ax.bar(x + w,   rbde_means,  w, color=COLOURS["RBDE"],   label="RBDE",   alpha=0.88, zorder=3)

    # Cohen's d annotations: DAPTA vs PPO
    for i in range(len(METRICS)):
        dapta_arr = np.array([rq2["DAPTA_vs_RBDE"][m]["dapta_mean"] for m in METRICS])
        diff = ppo_disc[:, i] - dapta_arr[i]
        std  = np.std(ppo_disc[:, i], ddof=1)
        d    = float(np.mean(ppo_disc[:, i] - dapta_arr[i]) / std) if std > 0 else 0.0
        ymax = max(dapta_means[i], ppo_means[i], rbde_means[i]) + 0.005
        ax.text(x[i], ymax, f"d={d:+.2f}",
                ha="center", fontsize=7, color="gray",
                fontfamily="monospace")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("Mean discourse improvement")
    ax.set_title(
        "Figure 7 — PPO vs DAPTA vs RBDE\n"
        "Comparing RL algorithm choice (DDQN vs PPO) against rule-based baseline",
        pad=12,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig7_ppo_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_rq4_aphasia_only(eval_dir: Path, out_dir: Path) -> None:
    """Figure 8 — RQ4 aphasia-only reanalysis."""
    path = eval_dir / "rq4_aphasia_only_results.json"
    if not path.exists():
        print("  Skipping fig8 — rq4_aphasia_only_results.json not found.")
        return

    with open(path) as f:
        data = json.load(f)

    pooled = data.get("pooled", {})
    if not pooled:
        print("  Skipping fig8 — no pooled data.")
        return

    # --- Load patient-level data for right panel ---
    NON_APHASIA = {
        "control", "Control", "CONTROL",
        "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
        "not_aphasic", "healthy",
    }

    dapta = np.load(eval_dir / "improvements_DAPTA.npy")
    gddqn = np.load(eval_dir / "improvements_G_DDQN.npy")

    # Load test patient profiles
    dae_dir = eval_dir.parent / "dae"
    dae_data        = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    all_session_ids = list(dae_data["session_ids"])
    sid_to_idx      = {sid: i for i, sid in enumerate(all_session_ids)}

    with open(dae_dir / "splits.json") as f:
        test_ids = json.load(f)["test"]
    with open(dae_dir / "patient_profiles.json") as f:
        all_profiles = json.load(f)

    test_indices  = [sid_to_idx[sid] for sid in test_ids if sid in sid_to_idx]
    test_profiles = [all_profiles[i] for i in test_indices]

    subtypes     = [p.get("aphasia_subtype", "Other") for p in test_profiles]
    aphasia_mask = np.array([s not in NON_APHASIA for s in subtypes])

    dapta_ap   = dapta[aphasia_mask]
    gddqn_ap   = gddqn[aphasia_mask]
    subtypes_ap = [s for s, m in zip(subtypes, aphasia_mask) if m]

    # Per-subtype CIU means
    unique_subtypes = sorted(set(subtypes_ap))
    subtype_dapta   = []
    subtype_gddqn   = []
    subtype_d       = []
    subtype_n       = []

    for st in unique_subtypes:
        mask  = np.array([s == st for s in subtypes_ap])
        d_arr = dapta_ap[mask, 0]
        g_arr = gddqn_ap[mask, 0]
        na, nb = len(d_arr), len(g_arr)
        pooled_std = np.sqrt(
            ((na - 1) * np.var(d_arr, ddof=1) + (nb - 1) * np.var(g_arr, ddof=1))
            / (na + nb - 2)
        ) if na + nb > 2 else 1.0
        d_val = float((np.mean(d_arr) - np.mean(g_arr)) / pooled_std) if pooled_std > 0 else 0.0
        subtype_dapta.append(float(np.mean(d_arr)))
        subtype_gddqn.append(float(np.mean(g_arr)))
        subtype_d.append(d_val)
        subtype_n.append(int(mask.sum()))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: pooled comparison
    ax = axes[0]
    dapta_mean   = pooled.get("dapta_ciu_mean", 0)
    gddqn_mean   = pooled.get("gddqn_ciu_mean", 0)
    d_val        = pooled.get("cohens_d", 0)
    p_val        = pooled.get("wilcoxon_p", 1)
    ci_lo, ci_hi = pooled.get("ci_95", [0, 0])

    ax.bar(
        ["DAPTA", "G-DDQN"],
        [dapta_mean, gddqn_mean],
        color=[COLOURS["DAPTA"], COLOURS["G_DDQN"]],
        alpha=0.88, width=0.4, zorder=3,
    )
    err_lo = max(dapta_mean - ci_lo, 0)
    err_hi = max(ci_hi - dapta_mean, 0)
    ax.errorbar(
        0, dapta_mean,
        yerr=[[err_lo], [err_hi]],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4,
    )
    ax.set_ylabel("Mean CIU rate improvement")
    ax.set_title(
        f"CIU Rate: DAPTA vs G-DDQN\n"
        f"(Aphasia-only, n={data.get('n_aphasia', '?')})\n"
        f"d = {d_val:.3f}  p = {p_val:.4f}"
        f"{'  *' if p_val < 0.05 else '  (n.s.)'}",
    )

    # Right: per-subtype breakdown
    ax2 = axes[1]
    x  = np.arange(len(unique_subtypes))
    w  = 0.35
    ax2.bar(x - w/2, subtype_dapta, w, color=COLOURS["DAPTA"],  label="DAPTA",  alpha=0.88, zorder=3)
    ax2.bar(x + w/2, subtype_gddqn, w, color=COLOURS["G_DDQN"], label="G-DDQN", alpha=0.88, zorder=3)

    for i, (d, n) in enumerate(zip(subtype_d, subtype_n)):
        ymax = max(subtype_dapta[i], subtype_gddqn[i]) + 0.005
        ax2.text(
            x[i], ymax, f"d={d:+.2f}",
            ha="center", fontsize=8, fontfamily="monospace",
            color=COLOURS["DAPTA"] if d > 0 else COLOURS["G_DDQN"],
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{st}\n(n={n})" for st, n in zip(unique_subtypes, subtype_n)],
        fontsize=8, rotation=15, ha="right",
    )
    ax2.set_ylabel("Mean CIU improvement")
    ax2.set_title("Per-subtype CIU\n(Aphasia patients only)")
    ax2.legend(fontsize=9)

    n_excluded = data.get("n_excluded", 0)
    n_total    = data.get("n_total", 0)
    n_aphasia  = data.get("n_aphasia", 0)
    fig.text(
        0.5, -0.02,
        f"Note: {n_excluded} non-aphasic patients excluded "
        f"({n_aphasia} aphasia patients retained from {n_total} total)",
        ha="center", fontsize=9, color="gray", style="italic",
    )

    fig.suptitle(
        "Figure 8 — RQ4 Reanalysis: Aphasia-Only Patients\n"
        "(Non-aphasic patients excluded at patient level)",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = out_dir / "fig8_rq4_aphasia_only.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot DAPTA thesis results")
    parser.add_argument("--eval_dir", default="outputs/evaluation")
    parser.add_argument("--out_dir",  default="outputs/figures")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from {eval_dir}/evaluation_results.json ...")
    results      = load_results(eval_dir)
    subtype_data = load_subtype(eval_dir)

    # Load PPO directly from saved array
    ppo_path = eval_dir / "improvements_PPO.npy"
    ppo_disc = np.load(str(ppo_path)) if ppo_path.exists() else None
    if ppo_disc is not None:
        print(f"  PPO improvements loaded: shape {ppo_disc.shape}")
    else:
        print("  PPO improvements not found — fig7 will be skipped.")

    print("\nGenerating figures:")
    plot_mean_improvement(results, out_dir)
    plot_cohens_d_heatmap(results, out_dir)
    plot_rq2(results, out_dir)
    plot_rq3(results, out_dir)
    plot_rq4(results, out_dir)
    plot_rq4_aphasia_only(eval_dir, out_dir)
    plot_subtype(subtype_data, out_dir)
    plot_ppo_comparison(results, ppo_disc, out_dir)

    print(f"\nAll figures saved to {out_dir}/")
    print("Done.")

if __name__ == "__main__":
    main()