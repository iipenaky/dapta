"""
Comprehensive thesis-ready plotting for the DAPTA system.
Covers all outputs from run_pes.py, run_rl.py, run_evaluation.py,
and run_sensitivity_analysis.py.

Usage:
    python plot_results.py

    # Or with custom output directories:
    python plot_results.py --pes_dir outputs/pes --rl_dir outputs/rl \
        --eval_dir outputs/evaluation --sens_dir outputs/sensitivity \
        --out_dir figures/

All figures are saved to figures/ (or --out_dir) as 300 DPI PDFs and PNGs.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# House style
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
})

# Palette — agent colours
AGENT_COLORS = {
    "DAPTA":              "#2166AC",
    "G_DDQN":             "#4DAC26",
    "NO_GRU_DAPTA":       "#74ADD1",
    "NO_GRU_G_DDQN":      "#A1D76A",
    "PPO_GENERALISED":    "#D6604D",
    "PPO_PERSONALISED":   "#F4A582",
    "RBDE":               "#636363",
    "RTS":                "#BDBDBD",
}

AGENT_LABELS = {
    "DAPTA":             "DAPTA (GRU, pers.)",
    "G_DDQN":            "G-DDQN (GRU, gen.)",
    "NO_GRU_DAPTA":      "No-GRU DAPTA",
    "NO_GRU_G_DDQN":     "No-GRU G-DDQN",
    "PPO_GENERALISED":   "PPO (gen.)",
    "PPO_PERSONALISED":  "PPO (pers.)",
    "RBDE":              "RBDE (clinical)",
    "RTS":               "Random",
}

METRIC_LABELS = {
    "ciu_rate":             "CIU Rate",
    "mc_score":             "Main Concept",
    "mlu_morphemes":        "MLU (morphemes)",
    "mattr":                "MATTR",
    "syntactic_complexity": "Syntactic Complexity",
}

METRIC_NAMES = list(METRIC_LABELS.keys())

NOISE_COLORS = {
    "low":    "#4575B4",
    "medium": "#FEE08B",
    "high":   "#D73027",
}
 
# Helpers
def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / f"{name}.pdf"))
    fig.savefig(str(out_dir / f"{name}.png"))
    plt.close(fig)
    print(f"  Saved: {name}")


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"  [WARN] Not found: {path}")
    return {}


def load_npy(path: Path) -> np.ndarray:
    if path.exists():
        return np.load(str(path))
    print(f"  [WARN] Not found: {path}")
    return np.array([])


def load_agent_arrays(arrays_dir: Path) -> dict:
    """Load all per-agent .npy arrays from an output directory."""
    agents = {}
    for f in sorted(arrays_dir.glob("*.npy")):
        name = f.stem.replace("test_", "").replace("train_improvements_", "")
        arr = np.load(str(f))
        if arr.size > 0:
            agents[name] = arr
    return agents


def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negl."
    if d < 0.5: return "small"
    if d < 0.8: return "med."
    return "large"


def bootstrap_ci(diff: np.ndarray, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# PES figures
def fig_cluster_distribution(pes_dir: Path, out_dir: Path):
    """Bar chart of patient counts per cluster."""
    data = load_json(pes_dir / "cluster_assignments.json")
    if not data:
        return
    sizes = {int(k): v for k, v in data.get("cluster_sizes", {}).items()}
    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    clusters = sorted(sizes)
    counts   = [sizes[c] for c in clusters]
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(clusters)))

    bars = ax.bar([f"C{c}" for c in clusters], counts, color=colors,
                  edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of training patients")
    ax.set_title("Patient cluster sizes (training split)")
    fig.tight_layout()
    save(fig, out_dir, "pes_cluster_distribution")


def fig_state_space_tsne(pes_dir: Path, out_dir: Path):
    """t-SNE projection of patient state vectors coloured by cluster."""
    npz = pes_dir / "env_initial_states.npz"
    if not npz.exists():
        return
    data           = np.load(str(npz), allow_pickle=True)
    states         = data["initial_states"]
    cluster_labels = data["cluster_labels"]
    split_labels   = data["split_labels"]

    n_clusters = int(cluster_labels.max()) + 1
    cmap       = plt.cm.tab10

    # t-SNE on first 38 dims (discourse block)
    X = states[:, :38]
    if len(X) > 1500:
        idx = np.random.choice(len(X), 1500, replace=False)
        X, cluster_labels, split_labels = X[idx], cluster_labels[idx], split_labels[idx]

    emb = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1)).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: coloured by cluster
    ax = axes[0]
    for c in range(n_clusters):
        mask = cluster_labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.6,
                   c=[cmap(c / max(n_clusters - 1, 1))], label=f"Cluster {c}")
    ax.set_title("t-SNE: patient state space by cluster")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.5, fontsize=8, ncol=2)

    # Right: coloured by split
    ax = axes[1]
    split_colors = {"train": "#2166AC", "val": "#F4A582", "test": "#D73027"}
    for split, col in split_colors.items():
        mask = split_labels == split
        if mask.sum() > 0:
            ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.6,
                       c=col, label=split.capitalize())
    ax.set_title("t-SNE: patient state space by split")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.5)

    fig.tight_layout()
    save(fig, out_dir, "pes_tsne_state_space")


def fig_transition_model_validation(pes_dir: Path, out_dir: Path):
    """MAE and directional accuracy per metric from validation."""
    data = load_json(pes_dir / "transition_model_validation.json")
    mae  = data.get("mae_per_metric", {})
    dacc = data.get("directional_accuracy", {})
    if not mae:
        return

    metrics = list(mae.keys())
    x       = np.arange(len(metrics))
    labels  = [METRIC_LABELS.get(m, m) for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    bars = ax.bar(x, [mae[m] for m in metrics],
                  color="#4575B4", alpha=0.8, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("MAE"); ax.set_title("Transition model: MAE per metric")
    for bar, m in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{mae[m]:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, [dacc.get(m, 0) for m in metrics],
                  color="#D73027", alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.50)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Directional accuracy"); ax.set_title("Transition model: directional accuracy")
    for bar, m in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{dacc.get(m, 0):.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend()

    fig.tight_layout()
    save(fig, out_dir, "pes_transition_model_validation")


def fig_cluster_aphasia_subtypes(pes_dir: Path, eval_dir: Path, out_dir: Path):
    """Stacked bar of aphasia subtypes per cluster from evaluation results."""
    data = load_json(eval_dir / "cluster_performance.json")
    rq4  = load_json(eval_dir / "test_rq4_personalisation.json")
    per_cluster = rq4.get("per_cluster", data)
    if not per_cluster:
        return

    clusters     = sorted(per_cluster.keys(), key=int)
    all_subtypes = set()
    for c in clusters:
        all_subtypes.update(per_cluster[c].get("subtype_counts", {}).keys())
    all_subtypes = sorted(all_subtypes)

    cmap   = plt.cm.Set2
    colors = {s: cmap(i / max(len(all_subtypes) - 1, 1)) for i, s in enumerate(all_subtypes)}

    fig, ax = plt.subplots(figsize=(9, 4))
    bottoms = np.zeros(len(clusters))
    for subtype in all_subtypes:
        vals = [per_cluster[c].get("subtype_counts", {}).get(subtype, 0) for c in clusters]
        ax.bar(clusters, vals, bottom=bottoms, label=subtype,
               color=colors[subtype], edgecolor="white", linewidth=0.6)
        bottoms += np.array(vals)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of patients")
    ax.set_title("Aphasia subtype composition per cluster (test split)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, out_dir, "pes_cluster_subtype_composition")

# Training curve figures
def _smooth(arr: list, w: int = 5) -> np.ndarray:
    """Simple moving-average smoother."""
    a = np.array(arr, dtype=float)
    if len(a) < w:
        return a
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="same")


def fig_training_curves(rl_dir: Path, out_dir: Path):
    """
    Three-panel training figure from training_logs.json.
    Handles the actual format: keys like 'dapta_cluster_0', 'g_ddqn', etc.,
    each containing 'steps', 'mean_reward', 'mean_ciu_improvement', 'loss'.
    Panel A — mean reward  (DAPTA clusters + G-DDQN)
    Panel B — mean CIU improvement
    Panel C — TD loss
    """
    logs = load_json(rl_dir / "training_logs.json")
    if not logs:
        return

    #  Classify every key 
    # skip PPO entries that only have {"status": ..., "steps": ...}
    def has_curves(val):
        return (isinstance(val, dict)
                and "mean_reward" in val
                and isinstance(val["mean_reward"], list)
                and len(val["mean_reward"]) > 0)

    dapta_keys   = sorted(k for k in logs if k.startswith("dapta_cluster_")  and has_curves(logs[k]))
    no_gru_keys  = sorted(k for k in logs if k.startswith("no_gru_dapta_")   and has_curves(logs[k]))
    gen_keys     = [k for k in ("g_ddqn", "no_gru_g_ddqn") if k in logs and has_curves(logs[k])]

    all_keys = dapta_keys + no_gru_keys + gen_keys
    if not all_keys:
        print("  [INFO] No reward curves found in training_logs.json — skipping")
        return

    #  Colour map 
    # DAPTA clusters: shades of blue; No-GRU clusters: shades of teal
    # Generalised agents: fixed colours from AGENT_COLORS
    cluster_blues = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(dapta_keys), 1)))
    cluster_teals = plt.cm.Greens(np.linspace(0.4, 0.9, max(len(no_gru_keys), 1)))

    def get_color(key, idx_d, idx_n):
        if key in AGENT_COLORS:
            return AGENT_COLORS[key.upper() if key.upper() in AGENT_COLORS else key]
        if key.startswith("dapta_cluster_"):
            return cluster_blues[idx_d]
        if key.startswith("no_gru_dapta_"):
            return cluster_teals[idx_n]
        return "#888888"

    def get_label(key):
        if key == "g_ddqn":      return "G-DDQN"
        if key == "no_gru_g_ddqn": return "No-GRU G-DDQN"
        if key.startswith("dapta_cluster_"):
            c = key.replace("dapta_cluster_", "")
            return f"DAPTA C{c}"
        if key.startswith("no_gru_dapta_cluster_"):
            c = key.replace("no_gru_dapta_cluster_", "")
            return f"No-GRU C{c}"
        return key

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("mean_reward",          axes[0], "Mean episode reward",    "A — Episode reward"),
        ("mean_ciu_improvement", axes[1], "Mean CIU improvement",   "B — CIU improvement"),
        ("loss",                 axes[2], "TD loss",                 "C — Training loss"),
    ]

    idx_d = idx_n = 0
    for key in all_keys:
        val   = logs[key]
        steps = val.get("steps", list(range(1, len(val["mean_reward"]) + 1)))
        col   = get_color(key, idx_d, idx_n)
        lbl   = get_label(key)
        lw    = 2.0 if key in ("g_ddqn", "no_gru_g_ddqn") else 1.2
        alpha = 1.0 if key in ("g_ddqn", "no_gru_g_ddqn") else 0.75
        ls    = "--" if key.startswith("no_gru") else "-"

        if key.startswith("dapta_cluster_"):  idx_d += 1
        if key.startswith("no_gru_dapta_"):   idx_n += 1

        for field, ax, _, __ in metrics:
            raw = val.get(field)
            if raw is None or len(raw) == 0:
                continue
            y = _smooth(raw, w=5)
            ax.plot(steps, y, color=col, linewidth=lw, alpha=alpha,
                    linestyle=ls, label=lbl)

    for field, ax, ylabel, title in metrics:
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
        )
        if field == "loss":
            ax.set_yscale("log")
        if field == "mean_reward":
            ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")

    # Single legend on the first axis (de-duplicate labels)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(),
                   fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("Training curves — all DAPTA agents (smoothed, window=5)", fontsize=12)
    fig.tight_layout()
    save(fig, out_dir, "rl_training_curves")


def fig_training_curves_ciu_per_cluster(rl_dir: Path, out_dir: Path):
    """
    Dedicated figure: CIU improvement curves per cluster, GRU vs No-GRU side by side.
    One subplot per cluster.
    """
    logs = load_json(rl_dir / "training_logs.json")
    if not logs:
        return

    def has_curves(val):
        return (isinstance(val, dict)
                and "mean_ciu_improvement" in val
                and isinstance(val["mean_ciu_improvement"], list)
                and len(val["mean_ciu_improvement"]) > 0)

    # find how many clusters
    cluster_ids = sorted(set(
        int(k.replace("dapta_cluster_", ""))
        for k in logs if k.startswith("dapta_cluster_") and has_curves(logs[k])
    ))
    if not cluster_ids:
        return

    ncols = min(3, len(cluster_ids))
    nrows = (len(cluster_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, c in enumerate(cluster_ids):
        ax  = axes[idx // ncols][idx % ncols]
        gru_key    = f"dapta_cluster_{c}"
        no_gru_key = f"no_gru_dapta_cluster_{c}"

        for key, col, lbl, ls in [
            (gru_key,    AGENT_COLORS["DAPTA"],       f"DAPTA C{c} (GRU)",    "-"),
            (no_gru_key, AGENT_COLORS["NO_GRU_DAPTA"], f"No-GRU C{c}", "--"),
        ]:
            if key not in logs or not has_curves(logs[key]):
                continue
            val   = logs[key]
            steps = val.get("steps", list(range(1, len(val["mean_ciu_improvement"]) + 1)))
            y     = _smooth(val["mean_ciu_improvement"], w=5)
            ax.plot(steps, y, color=col, linewidth=1.8, linestyle=ls, label=lbl)

        ax.set_title(f"Cluster {c}")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Mean CIU improvement")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
        )
        ax.legend(fontsize=8)

    # hide unused subplots
    for idx in range(len(cluster_ids), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("CIU improvement per cluster: GRU vs No-GRU DAPTA", fontsize=12)
    fig.tight_layout()
    save(fig, out_dir, "rl_training_ciu_per_cluster")

# RQ2 figures (DAPTA vs baselines)
def fig_rq2_grouped_bar(eval_dir: Path, out_dir: Path):
    """
    Grouped bar chart of mean CIU rate across all agents + baselines.
    Primary RQ2 figure.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    if not arrays_dir.exists():
        print("  [WARN] test_per_agent_arrays/ not found — trying train arrays")
        arrays_dir = Path("outputs/rl")

    agents = load_agent_arrays(arrays_dir)
    if not agents:
        return

    # Sort: pers agents first, then gen, then baselines
    order = ["DAPTA", "NO_GRU_DAPTA", "PPO_PERSONALISED",
             "G_DDQN", "NO_GRU_G_DDQN", "PPO_GENERALISED",
             "RBDE", "RTS"]
    plot_agents = [a for a in order if a in agents]

    ciu_means = []
    ciu_ses   = []
    for name in plot_agents:
        vals = agents[name][:, 0]
        ciu_means.append(np.mean(vals))
        ciu_ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    fig, ax = plt.subplots(figsize=(11, 5))
    x      = np.arange(len(plot_agents))
    colors = [AGENT_COLORS.get(a, "#888888") for a in plot_agents]

    bars = ax.bar(x, ciu_means, yerr=ciu_ses, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.2, "ecolor": "black"})

    # Significance stars vs RBDE
    if "RBDE" in agents:
        rbde_vals = agents["RBDE"][:, 0]
        for i, name in enumerate(plot_agents):
            if name in ("RBDE", "RTS"):
                continue
            vals = agents[name][:, 0]
            diff = vals - rbde_vals[:len(vals)]
            if np.all(diff == 0) or len(diff) < 2:
                continue
            try:
                _, p = scipy_stats.wilcoxon(diff, alternative="greater")
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if star:
                    ax.text(i, ciu_means[i] + ciu_ses[i] + 0.003, star,
                            ha="center", fontsize=11, color="black")
            except Exception:
                pass

    # Divider between personalised / generalised / baselines
    pers = [a for a in ["DAPTA", "NO_GRU_DAPTA", "PPO_PERSONALISED"] if a in plot_agents]
    gen  = [a for a in ["G_DDQN", "NO_GRU_G_DDQN", "PPO_GENERALISED"] if a in plot_agents]
    if pers and gen:
        ax.axvline(len(pers) - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    if gen:
        ax.axvline(len(pers) + len(gen) - 0.5, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS.get(a, a) for a in plot_agents],
                       rotation=25, ha="right")
    ax.set_ylabel("Mean CIU rate improvement")
    ax.set_title("RQ2: Agent performance vs baselines (test set, * p < 0.05 vs RBDE)")

    # Section labels
    if pers:
        ax.text(len(pers) / 2 - 0.5, ax.get_ylim()[1] * 0.97,
                "Personalised", ha="center", fontsize=9, color="#555555",
                style="italic")
    if gen:
        ax.text(len(pers) + len(gen) / 2 - 0.5, ax.get_ylim()[1] * 0.97,
                "Generalised", ha="center", fontsize=9, color="#555555",
                style="italic")

    fig.tight_layout()
    save(fig, out_dir, "rq2_agent_comparison_bar")


def fig_rq2_radar(eval_dir: Path, out_dir: Path):
    """Radar (spider) chart comparing DAPTA, G-DDQN, and RBDE across 5 metrics."""
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents     = load_agent_arrays(arrays_dir)
    if not agents:
        return

    plot_agents = [a for a in ["DAPTA", "G_DDQN", "RBDE"] if a in agents]
    if len(plot_agents) < 2:
        return

    metrics = METRIC_NAMES[:5]
    means   = {a: [np.mean(agents[a][:, i]) for i in range(5)] for a in plot_agents}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for name in plot_agents:
        vals = means[name] + means[name][:1]
        ax.plot(angles, vals, linewidth=2, color=AGENT_COLORS[name],
                label=AGENT_LABELS[name])
        ax.fill(angles, vals, alpha=0.1, color=AGENT_COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_title("RQ2: Discourse metrics — radar comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "rq2_radar_metrics")


def fig_rq2_cohens_d_heatmap(eval_dir: Path, out_dir: Path):
    """Heatmap of Cohen's d (agent vs RBDE) across all agents × 5 metrics."""
    rq2 = load_json(eval_dir / "test_rq2_results.json")
    per_agent = rq2.get("per_agent", {})
    if not per_agent:
        return

    agents  = [a for a in AGENT_COLORS if a not in ("RBDE", "RTS") and a in per_agent]
    metrics = METRIC_NAMES

    matrix = np.zeros((len(agents), len(metrics)))
    sig    = np.zeros((len(agents), len(metrics)), dtype=bool)

    for i, agent in enumerate(agents):
        key = f"{agent}_vs_RBDE"
        entry = per_agent[agent].get(key, {})
        for j, metric in enumerate(metrics):
            m = entry.get(metric, {})
            matrix[i, j] = m.get("cohens_d", 0.0)
            sig[i, j]    = m.get("significant", False)

    fig, ax = plt.subplots(figsize=(9, max(3, len(agents) * 0.65)))
    vmax = max(abs(matrix.max()), abs(matrix.min()), 0.8)
    im   = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=25, ha="right")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([AGENT_LABELS.get(a, a) for a in agents])
    ax.set_title("RQ2: Cohen's d vs RBDE — all agents × metrics\n(* = Bonferroni-corrected p < 0.05)")

    for i in range(len(agents)):
        for j in range(len(metrics)):
            star = "*" if sig[i, j] else ""
            ax.text(j, i, f"{matrix[i, j]:.2f}{star}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(matrix[i, j]) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.6)
    fig.tight_layout()
    save(fig, out_dir, "rq2_cohens_d_heatmap")


def fig_rq2_effect_sizes_with_ci(eval_dir: Path, out_dir: Path):
    """Forest plot: Cohen's d + 95% CI for DAPTA vs RBDE, one row per metric."""
    rq2 = load_json(eval_dir / "test_rq2_results.json")
    per_agent = rq2.get("per_agent", {})
    if "DAPTA" not in per_agent:
        return

    entry = per_agent["DAPTA"].get("DAPTA_vs_RBDE", {})
    metrics = METRIC_NAMES
    ds   = [entry.get(m, {}).get("cohens_d", 0.0) for m in metrics]
    cis  = [entry.get(m, {}).get("ci_95", [0, 0]) for m in metrics]
    sigs = [entry.get(m, {}).get("significant", False) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(metrics))

    for i, (d, ci, sig) in enumerate(zip(ds, cis, sigs)):
        color = "#2166AC" if sig else "#AAAAAA"
        lo, hi = float(ci[0]), float(ci[1])
        # ci_95 stores [low_bound, high_bound] of the *difference* distribution.
        # Convert to half-widths for xerr; clamp to 0 so matplotlib never sees negatives.
        xerr_lo = max(d - lo, 0.0)
        xerr_hi = max(hi - d, 0.0)
        ax.errorbar(d, y[i], xerr=[[xerr_lo], [xerr_hi]],
                    fmt="o", color=color, capsize=5, markersize=6,
                    elinewidth=1.5, capthick=1.5)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0.4, color="#D73027", linewidth=0.8, linestyle=":", alpha=0.6,
               label="Clinical threshold (d=0.40)")
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_xlabel("Cohen's d (DAPTA vs RBDE)\n[blue = Bonferroni-significant]")
    ax.set_title("RQ2: DAPTA effect sizes + 95% bootstrap CI")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "rq2_forest_plot")


# RQ3 figures (cross-task transfer)
def fig_rq3_transfer_scatter(eval_dir: Path, out_dir: Path):
    """Scatter: structured task CIU gain vs conversation CIU gain."""
    csv_path = eval_dir / "dapta_test_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(str(csv_path))
    df = df.dropna(subset=["structured_ciu_gain", "conversation_ciu_gain"])
    if len(df) < 5:
        return

    r, p = scipy_stats.spearmanr(df["structured_ciu_gain"], df["conversation_ciu_gain"])

    fig, ax = plt.subplots(figsize=(6, 5))

    # Color by cluster if available
    if "cluster_id" in df.columns:
        clusters = df["cluster_id"].unique()
        cmap     = plt.cm.tab10
        for c in sorted(clusters):
            mask = df["cluster_id"] == c
            ax.scatter(df.loc[mask, "structured_ciu_gain"],
                       df.loc[mask, "conversation_ciu_gain"],
                       s=40, alpha=0.7, label=f"Cluster {c}",
                       color=cmap(c / max(len(clusters) - 1, 1)))
        ax.legend(fontsize=8, markerscale=1.2)
    else:
        ax.scatter(df["structured_ciu_gain"], df["conversation_ciu_gain"],
                   s=40, alpha=0.7, color="#2166AC")

    # Regression line
    m_fit, b_fit = np.polyfit(df["structured_ciu_gain"], df["conversation_ciu_gain"], 1)
    xs = np.linspace(df["structured_ciu_gain"].min(), df["structured_ciu_gain"].max(), 100)
    ax.plot(xs, m_fit * xs + b_fit, color="#D73027", linewidth=1.5,
            linestyle="--", label=f"Fit: ρ={r:.3f}, p={p:.3f}")
    ax.axhline(0, color="gray", linewidth=0.5); ax.axvline(0, color="gray", linewidth=0.5)

    ax.set_xlabel("Structured task CIU gain (mean of structured tasks)")
    ax.set_ylabel("Conversation CIU gain")
    ax.set_title(f"RQ3: Cross-task transfer (Spearman ρ = {r:.3f}, p = {p:.3f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "rq3_transfer_scatter")


def fig_rq3_task_gains_violin(eval_dir: Path, out_dir: Path):
    """Violin plots of structured vs conversation CIU gains."""
    csv_path = eval_dir / "dapta_test_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(str(csv_path))
    df = df.dropna(subset=["structured_ciu_gain", "conversation_ciu_gain"])
    if len(df) < 5:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    data   = [df["structured_ciu_gain"].values, df["conversation_ciu_gain"].values]
    labels = ["Structured tasks", "Conversation"]
    colors = ["#2166AC", "#D73027"]

    parts = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col); pc.set_alpha(0.5)

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylabel("CIU gain within DAPTA episode")
    ax.set_title("RQ3: Distribution of task-level CIU gains")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    save(fig, out_dir, "rq3_task_gains_violin")


# RQ4 figures (personalisation)
def fig_rq4_personalised_vs_generalised(eval_dir: Path, out_dir: Path):
    """
    Side-by-side boxplots of DAPTA vs G-DDQN CIU rate across all test patients.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents     = load_agent_arrays(arrays_dir)
    dapta = agents.get("DAPTA"); gddqn = agents.get("G_DDQN")
    if dapta is None or gddqn is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box/strip plot
    ax = axes[0]
    data   = [dapta[:, 0], gddqn[:, 0]]
    bx = ax.boxplot(data, patch_artist=True, notch=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, col in zip(bx["boxes"], [AGENT_COLORS["DAPTA"], AGENT_COLORS["G_DDQN"]]):
        patch.set_facecolor(col); patch.set_alpha(0.6)

    n = min(len(dapta), len(gddqn), 200)
    jitter = 0.12
    ax.scatter(np.ones(n) + np.random.uniform(-jitter, jitter, n),
               dapta[:n, 0], s=8, alpha=0.4, color=AGENT_COLORS["DAPTA"])
    ax.scatter(np.ones(n) * 2 + np.random.uniform(-jitter, jitter, n),
               gddqn[:n, 0], s=8, alpha=0.4, color=AGENT_COLORS["G_DDQN"])

    # Paired lines for first 50 patients
    n_lines = min(50, len(dapta), len(gddqn))
    for i in range(n_lines):
        ax.plot([1, 2], [dapta[i, 0], gddqn[i, 0]],
                color="gray", alpha=0.1, linewidth=0.5)

    _, p = scipy_stats.wilcoxon(dapta[:, 0] - gddqn[:len(dapta), 0], alternative="greater")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([AGENT_LABELS["DAPTA"], AGENT_LABELS["G_DDQN"]])
    ax.set_ylabel("CIU rate improvement")
    ax.set_title(f"RQ4: Personalised vs generalised\n(Wilcoxon p = {p:.4f})")

    # Per-metric Cohen's d comparison
    ax = axes[1]
    ds_pers = []
    ds_gen  = []
    rbde    = agents.get("RBDE")
    for i in range(5):
        if rbde is not None:
            ds_pers.append(np.mean(dapta[:, i]) - np.mean(rbde[:, i]))
            ds_gen.append( np.mean(gddqn[:, i]) - np.mean(rbde[:, i]))
        else:
            ds_pers.append(np.mean(dapta[:, i]))
            ds_gen.append( np.mean(gddqn[:, i]))

    x     = np.arange(5)
    width = 0.35
    ax.bar(x - width / 2, ds_pers, width, color=AGENT_COLORS["DAPTA"],
           alpha=0.8, label=AGENT_LABELS["DAPTA"], edgecolor="white")
    ax.bar(x + width / 2, ds_gen,  width, color=AGENT_COLORS["G_DDQN"],
           alpha=0.8, label=AGENT_LABELS["G_DDQN"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_NAMES[:5]],
                       rotation=20, ha="right")
    ylabel = "Mean improvement over RBDE" if rbde is not None else "Mean improvement"
    ax.set_ylabel(ylabel)
    ax.set_title("RQ4: Per-metric comparison")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.6)

    fig.tight_layout()
    save(fig, out_dir, "rq4_personalised_vs_generalised")


def fig_rq4_per_cluster(eval_dir: Path, out_dir: Path):
    """Per-cluster CIU rates: DAPTA vs G-DDQN."""
    data    = load_json(eval_dir / "cluster_performance.json")
    rq4     = load_json(eval_dir / "test_rq4_personalisation.json")
    per_cls = rq4.get("per_cluster", data)
    if not per_cls:
        return

    clusters = sorted(per_cls.keys(), key=int)
    d_ciu    = [per_cls[c].get("personalised_ciu_mean", per_cls[c].get("dapta_ciu_mean", 0))
                for c in clusters]
    g_ciu    = [per_cls[c].get("generalised_ciu_mean",  per_cls[c].get("gddqn_ciu_mean", 0))
                for c in clusters]
    ns       = [per_cls[c].get("n", 1) for c in clusters]

    x     = np.arange(len(clusters))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars_d = ax.bar(x - width / 2, d_ciu, width, color=AGENT_COLORS["DAPTA"],
                    alpha=0.8, label=AGENT_LABELS["DAPTA"], edgecolor="white")
    bars_g = ax.bar(x + width / 2, g_ciu, width, color=AGENT_COLORS["G_DDQN"],
                    alpha=0.8, label=AGENT_LABELS["G_DDQN"], edgecolor="white")

    for bar, n in zip(bars_d, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.002,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_ylabel("Mean CIU rate improvement")
    ax.set_title("RQ4: Per-cluster CIU rate — personalised vs generalised")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.6)
    fig.tight_layout()
    save(fig, out_dir, "rq4_per_cluster_bar")


def fig_rq4_wabaq_moderation(eval_dir: Path, out_dir: Path):
    """Scatter: WAB-AQ vs personalisation benefit (CIU DAPTA - CIU G-DDQN)."""
    csv_path = eval_dir / "dapta_test_results.csv"
    rq4      = load_json(eval_dir / "test_rq4_personalisation.json")
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents   = load_agent_arrays(arrays_dir)

    dapta = agents.get("DAPTA"); gddqn = agents.get("G_DDQN")
    if dapta is None or gddqn is None:
        return

    # Try to get WAB-AQ from profiles
    try:
        with open("outputs/dae/patient_profiles.json") as f:
            profiles = json.load(f)
    except Exception:
        print("  [WARN] patient_profiles.json not found — skipping WAB-AQ moderation plot")
        return

    # Match to test patients via session IDs from CSV
    if not csv_path.exists():
        return
    df = pd.read_csv(str(csv_path))
    if "session_id" not in df.columns:
        return

    session_to_wab = {str(p["session_id"]): float(p.get("wab_aq") or 55.0)
                      for p in profiles}
    wab_aqs = np.array([session_to_wab.get(str(sid), 55.0) for sid in df["session_id"]])
    n = min(len(wab_aqs), len(dapta), len(gddqn))
    benefit = dapta[:n, 0] - gddqn[:n, 0]
    wab_aqs = wab_aqs[:n]

    r, p = scipy_stats.pearsonr(wab_aqs, benefit)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(wab_aqs, benefit, s=30, alpha=0.6, color="#4575B4")
    m_fit, b_fit = np.polyfit(wab_aqs, benefit, 1)
    xs = np.linspace(wab_aqs.min(), wab_aqs.max(), 100)
    ax.plot(xs, m_fit * xs + b_fit, color="#D73027", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("WAB-AQ score")
    ax.set_ylabel("Personalisation benefit\n(DAPTA CIU − G-DDQN CIU)")
    ax.set_title(f"RQ4: WAB-AQ as moderator\n(Pearson r = {r:.3f}, p = {p:.4f})")

    # Subgroup bar: low vs high WAB-AQ
    ax = axes[1]
    low_mask  = wab_aqs <  50
    high_mask = wab_aqs >= 50
    subgroups  = []
    means_d, means_g, labels_sg = [], [], []
    for mask, lbl in [(low_mask, "Low WAB-AQ\n(< 50)"),
                      (high_mask, "High WAB-AQ\n(≥ 50)")]:
        if mask.sum() > 0:
            means_d.append(np.mean(dapta[mask, 0]))
            means_g.append(np.mean(gddqn[mask, 0]))
            labels_sg.append(lbl)

    x = np.arange(len(labels_sg)); width = 0.35
    ax.bar(x - width / 2, means_d, width, color=AGENT_COLORS["DAPTA"],
           alpha=0.8, label=AGENT_LABELS["DAPTA"], edgecolor="white")
    ax.bar(x + width / 2, means_g, width, color=AGENT_COLORS["G_DDQN"],
           alpha=0.8, label=AGENT_LABELS["G_DDQN"], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels_sg)
    ax.set_ylabel("Mean CIU rate improvement")
    ax.set_title("RQ4: Severity subgroup analysis")
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "rq4_wabaq_moderation")

# Ablation figures
def fig_ablation_summary(eval_dir: Path, out_dir: Path):
    """
    Grouped bar: CIU Cohen's d vs RBDE for every ablated agent.
    Clearly shows GRU benefit + algorithm comparison.
    """
    abl = load_json(eval_dir / "test_ablation_results.json")
    per = abl.get("per_agent_summary", {})
    if not per:
        return

    order = ["DAPTA", "NO_GRU_DAPTA", "PPO_PERSONALISED",
             "G_DDQN", "NO_GRU_G_DDQN", "PPO_GENERALISED"]
    agents  = [a for a in order if a in per]
    d_vals  = [per[a].get("ciu_cohens_d_vs_rbde", 0.0) for a in agents]
    effects = [effect_label(d) for d in d_vals]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = [AGENT_COLORS.get(a, "#888888") for a in agents]
    bars = ax.bar(range(len(agents)), d_vals, color=colors,
                  edgecolor="white", linewidth=0.8, alpha=0.85)

    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axhline(0.4, color="#D73027", linestyle=":", linewidth=1,
               label="Clinical threshold (d=0.40)", alpha=0.7)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([AGENT_LABELS.get(a, a) for a in agents],
                       rotation=20, ha="right")
    ax.set_ylabel("Cohen's d (CIU vs RBDE)")
    ax.set_title("Ablation: GRU vs no-GRU, DDQN vs PPO")
    ax.legend(fontsize=9)

    for i, (bar, eff) in enumerate(zip(bars, effects)):
        y_pos = bar.get_height() + 0.01 if bar.get_height() >= 0 else bar.get_height() - 0.04
        ax.text(i, y_pos, eff, ha="center", fontsize=8, color="#333333")

    fig.tight_layout()
    save(fig, out_dir, "ablation_summary_bar")


def fig_ablation_gru_effect(eval_dir: Path, out_dir: Path):
    """
    Paired comparison: GRU vs No-GRU within personalised and generalised settings.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents     = load_agent_arrays(arrays_dir)

    pairs = [
        ("DAPTA",  "NO_GRU_DAPTA",  "Personalised (cluster-specific)"),
        ("G_DDQN", "NO_GRU_G_DDQN", "Generalised (pooled)"),
    ]
    pairs = [(a, b, lbl) for a, b, lbl in pairs if a in agents and b in agents]
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
    for ax, (gru_key, no_gru_key, title) in zip(axes[0], pairs):
        gru_vals    = agents[gru_key][:, 0]
        no_gru_vals = agents[no_gru_key][:, 0]
        data = [gru_vals, no_gru_vals]

        bx = ax.boxplot(data, patch_artist=True, notch=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        colors = [AGENT_COLORS[gru_key], AGENT_COLORS[no_gru_key]]
        for patch, col in zip(bx["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.6)

        n = min(80, len(gru_vals), len(no_gru_vals))
        jitter = 0.10
        ax.scatter(np.ones(n) + np.random.uniform(-jitter, jitter, n),
                   gru_vals[:n], s=8, alpha=0.5, color=AGENT_COLORS[gru_key])
        ax.scatter(np.ones(n) * 2 + np.random.uniform(-jitter, jitter, n),
                   no_gru_vals[:n], s=8, alpha=0.5, color=AGENT_COLORS[no_gru_key])

        diff = gru_vals - no_gru_vals[:len(gru_vals)]
        if np.any(diff != 0) and len(diff) >= 2:
            try:
                _, p = scipy_stats.wilcoxon(diff)
            except Exception:
                p = 1.0
        else:
            p = 1.0

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["GRU", "No-GRU"])
        ax.set_ylabel("CIU rate improvement")
        ax.set_title(f"GRU benefit — {title}\n(Wilcoxon p = {p:.4f})")

    fig.tight_layout()
    save(fig, out_dir, "ablation_gru_effect")

# Sensitivity analysis figures
def fig_sensitivity_overview(sens_dir: Path, out_dir: Path):
    """
    Multi-panel: how key statistics change across noise levels.
    """
    data = load_json(sens_dir / "sensitivity_results.json")
    summaries = data.get("summaries", [])
    if not summaries:
        # Fall back to CSV
        csv = sens_dir / "sensitivity_summary.csv"
        if csv.exists():
            df = pd.read_csv(str(csv))
            summaries_from_csv = df.to_dict("records")
        else:
            return

    # Extract from nested structure
    try:
        noise_labels = [s["noise_label"]  for s in summaries]
        noise_stds   = [s["noise_std"]    for s in summaries]
        rq2_d  = [s.get("rq2", {}).get("cohens_d_dapta_vs_rbde",  None) for s in summaries]
        rq2_p  = [s.get("rq2", {}).get("wilcoxon_p",              None) for s in summaries]
        rq4_d  = [s.get("rq4", {}).get("cohens_d_dapta_vs_gddqn", None) for s in summaries]
        rq3_r  = [s.get("rq3", {}).get("spearman_r",              None) for s in summaries]
        rq2_beats = [s.get("rq2", {}).get("dapta_beats_rbde",     False) for s in summaries]
        rq4_pers  = [s.get("rq4", {}).get("personalisation_beneficial", False) for s in summaries]

        dapta_ciu = [s.get("rq2", {}).get("dapta_ciu_mean", None) for s in summaries]
        rbde_ciu  = [s.get("rq2", {}).get("rbde_ciu_mean",  None) for s in summaries]
    except Exception as e:
        print(f"  [WARN] Sensitivity parse error: {e}")
        return

    x = np.arange(len(noise_labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Cohen's d for RQ2 and RQ4
    ax = axes[0, 0]
    if any(v is not None for v in rq2_d):
        ax.plot(x, [v for v in rq2_d], "o-", color="#2166AC",
                linewidth=2, label="RQ2: DAPTA vs RBDE")
    if any(v is not None for v in rq4_d):
        ax.plot(x, [v for v in rq4_d], "s--", color="#D73027",
                linewidth=2, label="RQ4: pers. vs gen.")
    ax.axhline(0.4, color="gray", linestyle=":", alpha=0.6, label="Clinical threshold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(noise_labels)
    ax.set_ylabel("Cohen's d"); ax.set_title("Effect sizes across noise levels")
    ax.legend(fontsize=9)

    # Panel 2: p-values
    ax = axes[0, 1]
    if any(v is not None for v in rq2_p):
        ax.plot(x, [v for v in rq2_p], "o-", color="#2166AC",
                linewidth=2, label="RQ2 p-value (Wilcoxon)")
    ax.axhline(0.05, color="#D73027", linestyle="--", label="α = 0.05")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(noise_labels)
    ax.set_ylabel("p-value (log scale)"); ax.set_title("Statistical significance across noise levels")
    ax.legend(fontsize=9)

    # Panel 3: absolute CIU means
    ax = axes[1, 0]
    if any(v is not None for v in dapta_ciu):
        ax.bar(x - 0.2, [v or 0 for v in dapta_ciu], 0.35,
               color=AGENT_COLORS["DAPTA"], alpha=0.8,
               label=AGENT_LABELS["DAPTA"], edgecolor="white")
    if any(v is not None for v in rbde_ciu):
        ax.bar(x + 0.2, [v or 0 for v in rbde_ciu], 0.35,
               color=AGENT_COLORS["RBDE"], alpha=0.8,
               label=AGENT_LABELS["RBDE"], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(noise_labels)
    ax.set_ylabel("Mean CIU rate improvement"); ax.set_title("Absolute CIU means across noise levels")
    ax.legend(fontsize=9)

    # Panel 4: RQ3 Spearman r
    ax = axes[1, 1]
    valid_r3 = [v for v in rq3_r if v is not None]
    if valid_r3:
        ax.plot(x[:len(valid_r3)], valid_r3, "D-", color="#4DAC26",
                linewidth=2, label="RQ3: Spearman ρ (transfer)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="Weak threshold (r=0.2)")
    ax.set_xticks(x); ax.set_xticklabels(noise_labels)
    ax.set_ylabel("Spearman ρ"); ax.set_title("Cross-task transfer correlation across noise levels")
    ax.legend(fontsize=9)

    fig.suptitle("Sensitivity analysis: key statistics across transition model noise levels",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "sensitivity_overview")


def fig_sensitivity_stability_table(sens_dir: Path, out_dir: Path):
    """
    Heatmap-style stability table: noise level × statistic.
    Each cell: value + background colour (green = good, red = bad).
    """
    csv = sens_dir / "sensitivity_summary.csv"
    if not csv.exists():
        return
    df = pd.read_csv(str(csv))
    if df.empty:
        return

    cols = {
        "rq2_cohens_d":    "RQ2 Cohen's d",
        "rq2_significant": "RQ2 sig.",
        "rq2_answered":    "RQ2 answered",
        "rq4_cohens_d":    "RQ4 Cohen's d",
        "rq4_significant": "RQ4 sig.",
        "rq4_pers_beneficial": "RQ4 pers. wins",
        "rq3_spearman_r":  "RQ3 Spearman ρ",
        "rq3_significant": "RQ3 sig.",
        "rq3_answered":    "RQ3 answered",
    }
    cols = {k: v for k, v in cols.items() if k in df.columns}
    if not cols:
        return

    cell_data = df[[c for c in cols]].copy()
    row_labels = df["noise_label"].tolist()
    col_labels = list(cols.values())

    # Build numeric matrix for color coding
    num_mat = np.zeros((len(cell_data), len(cols)))
    for j, col in enumerate(cols):
        vals = cell_data[col]
        try:
            num_mat[:, j] = pd.to_numeric(vals, errors="coerce").fillna(0).values
        except Exception:
            num_mat[:, j] = 0

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.4), max(3, len(row_labels) * 1.2)))
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(row_labels))
    ax.axis("off")
    ax.set_title("Sensitivity analysis: stability table\n(green = positive / significant, red = negative / not significant)",
                 fontsize=11)

    for i, row_label in enumerate(row_labels):
        for j, col in enumerate(cols):
            val  = cell_data.iloc[i][col]
            num  = num_mat[i, j]
            # Color logic
            if isinstance(val, bool) or str(val).lower() in ("true", "false"):
                is_true = str(val).lower() == "true"
                bg = "#C7E9C0" if is_true else "#FCBBA1"
                txt = str(val)
            else:
                try:
                    num_val = float(val)
                    bg = "#C7E9C0" if num_val > 0 else "#FCBBA1"
                    txt = f"{num_val:.3f}"
                except Exception:
                    bg = "#FFFFFF"; txt = str(val)

            rect = mpatches.FancyBboxPatch(
                (j + 0.05, i + 0.1), 0.88, 0.78,
                boxstyle="round,pad=0.02", facecolor=bg,
                edgecolor="#AAAAAA", linewidth=0.5, transform=ax.transData
            )
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=9, color="#222222")

    # Column headers
    for j, lbl in enumerate(col_labels):
        ax.text(j + 0.5, len(row_labels) + 0.1, lbl, ha="center", va="bottom",
                fontsize=9, fontweight="bold", rotation=20)
    # Row labels
    for i, lbl in enumerate(row_labels):
        ax.text(-0.1, i + 0.5, lbl, ha="right", va="center",
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    save(fig, out_dir, "sensitivity_stability_table")

# Summary / overview figures
def fig_all_agents_metric_profile(eval_dir: Path, out_dir: Path):
    """
    Line chart: all agents' mean improvement across the 5 discourse metrics.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents     = load_agent_arrays(arrays_dir)
    if not agents:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(5)
    for name, arr in agents.items():
        if arr.shape[1] < 5:
            continue
        means  = [np.mean(arr[:, i]) for i in range(5)]
        color  = AGENT_COLORS.get(name, "#888888")
        lw     = 2.5 if name in ("DAPTA", "RBDE") else 1.2
        ls     = "-" if name in ("DAPTA", "G_DDQN", "RBDE") else "--"
        alpha  = 1.0 if name in ("DAPTA", "G_DDQN", "RBDE") else 0.6
        ax.plot(x, means, "o" + ls, color=color, linewidth=lw, alpha=alpha,
                label=AGENT_LABELS.get(name, name), markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_NAMES[:5]], rotation=15, ha="right")
    ax.set_ylabel("Mean improvement")
    ax.set_title("All agents: discourse metric profile (test set)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, "overview_all_agents_metric_profile")


def fig_rq_summary_scorecard(eval_dir: Path, out_dir: Path):
    """
    Single-figure 'scorecard' for the thesis: RQ2, RQ3, RQ4 answered?
    """
    rq2 = load_json(eval_dir / "test_rq2_results.json")
    rq3 = load_json(eval_dir / "test_rq3_transfer.json")
    rq4 = load_json(eval_dir / "test_rq4_personalisation.json")

    rows = [
        ("RQ2", "Outperform clinical baseline?",
         rq2.get("rq2_answered_positively", False),
         f"Best CIU d: {_best_d_rq2(rq2)}"),
        ("RQ3", "Structured→conversation transfer?",
         rq3.get("rq3_answered_positively", False),
         f"ρ = {rq3.get('spearman_r', 'N/A')}, p = {rq3.get('p_value', 'N/A')}"),
        ("RQ4", "Personalisation > generalisation?",
         rq4.get("rq4_answered_positively", rq4.get("personalisation_beneficial", False)),
         f"CIU var. reduction: {rq4.get('ciu_variance_reduction_pct', 'N/A')}%"),
    ]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")

    for i, (rq_label, question, answered, detail) in enumerate(rows):
        y = 0.75 - i * 0.32
        color  = "#2CA02C" if answered else "#D62728"
        symbol = "✓" if answered else "✗"

        ax.text(0.01, y, rq_label, ha="left", va="center",
                fontsize=14, fontweight="bold", color="#333333",
                transform=ax.transAxes)
        ax.text(0.08, y, question, ha="left", va="center",
                fontsize=11, color="#555555", transform=ax.transAxes)
        ax.text(0.65, y, symbol, ha="center", va="center",
                fontsize=20, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.70, y, detail, ha="left", va="center",
                fontsize=10, color="#777777", transform=ax.transAxes,
                style="italic")

    ax.set_title("DAPTA thesis: research question scorecard (test set)",
                 fontsize=12, pad=12)
    fig.tight_layout()
    save(fig, out_dir, "rq_scorecard")


def _best_d_rq2(rq2: dict) -> str:
    tbl = rq2.get("summary_table", {})
    best_d = max(
        (row.get("ciu_cohens_d_vs_rbde", 0)
         for name, row in tbl.items() if row.get("type") != "baseline"),
        default=0
    )
    return f"{best_d:.3f}"


# Discourse improvement distributions
def fig_ciu_improvement_distributions(eval_dir: Path, out_dir: Path):
    """
    Overlapping KDE plots of per-patient CIU improvement for all agents.
    """
    from scipy.stats import gaussian_kde

    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents     = load_agent_arrays(arrays_dir)
    if not agents:
        return

    focus = ["DAPTA", "G_DDQN", "RBDE", "RTS"]
    plot_agents = [a for a in focus if a in agents]
    if not plot_agents:
        plot_agents = list(agents.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in plot_agents:
        vals = agents[name][:, 0]
        if len(vals) < 5:
            continue
        kde  = gaussian_kde(vals, bw_method="scott")
        xs   = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 300)
        lw   = 2.5 if name in ("DAPTA",) else 1.5
        ls   = "-" if name not in ("RBDE", "RTS") else "--"
        ax.plot(xs, kde(xs), color=AGENT_COLORS[name], linewidth=lw, linestyle=ls,
                label=AGENT_LABELS[name])
        ax.fill_between(xs, kde(xs), alpha=0.08, color=AGENT_COLORS[name])
        ax.axvline(np.mean(vals), color=AGENT_COLORS[name], linewidth=0.8,
                   linestyle=":", alpha=0.7)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("CIU rate improvement")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of per-patient CIU improvements (test set)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "ciu_improvement_distributions")


def fig_pairwise_improvement_scatter(eval_dir: Path, out_dir: Path):
    """
    Scatter of DAPTA vs G-DDQN per-patient CIU improvement.
    Points above the diagonal = personalisation helps that patient.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents     = load_agent_arrays(arrays_dir)
    dapta = agents.get("DAPTA"); gddqn = agents.get("G_DDQN")
    if dapta is None or gddqn is None:
        return

    n = min(len(dapta), len(gddqn))
    d_ciu = dapta[:n, 0]
    g_ciu = gddqn[:n, 0]

    pct_above = np.mean(d_ciu > g_ciu) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(g_ciu, d_ciu, s=20, alpha=0.5, color="#4575B4")
    mn, mx = min(g_ciu.min(), d_ciu.min()), max(g_ciu.max(), d_ciu.max())
    ax.plot([mn, mx], [mn, mx], color="gray", linewidth=1, linestyle="--",
            label="Equal performance")
    ax.set_xlabel(f"G-DDQN CIU improvement")
    ax.set_ylabel(f"DAPTA CIU improvement")
    ax.set_title(f"DAPTA vs G-DDQN per patient\n({pct_above:.1f}% benefit from personalisation)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "rq4_pairwise_improvement_scatter")

# Main
def main(args):
    pes_dir  = Path(args.pes_dir)
    rl_dir   = Path(args.rl_dir)
    eval_dir = Path(args.eval_dir)
    sens_dir = Path(args.sens_dir)
    out_dir  = Path(args.out_dir)

    print(f"\n{'='*60}")
    print("DAPTA thesis plotting — generating all figures")
    print(f"{'='*60}\n")

    #  PES figures 
    print("PES figures:")
    fig_cluster_distribution(pes_dir, out_dir / "pes")
    fig_state_space_tsne(pes_dir, out_dir / "pes")
    fig_transition_model_validation(pes_dir, out_dir / "pes")
    fig_cluster_aphasia_subtypes(pes_dir, eval_dir, out_dir / "pes")

    #  Training figures 
    print("\nTraining figures:")
    fig_training_curves(rl_dir, out_dir / "training")
    fig_training_curves_ciu_per_cluster(rl_dir, out_dir / "training")

    #  RQ2 figures 
    print("\nRQ2 figures:")
    fig_rq2_grouped_bar(eval_dir, out_dir / "rq2")
    fig_rq2_radar(eval_dir, out_dir / "rq2")
    fig_rq2_cohens_d_heatmap(eval_dir, out_dir / "rq2")
    fig_rq2_effect_sizes_with_ci(eval_dir, out_dir / "rq2")

    #  RQ3 figures 
    print("\nRQ3 figures:")
    fig_rq3_transfer_scatter(eval_dir, out_dir / "rq3")
    fig_rq3_task_gains_violin(eval_dir, out_dir / "rq3")

    #  RQ4 figures 
    print("\nRQ4 figures:")
    fig_rq4_personalised_vs_generalised(eval_dir, out_dir / "rq4")
    fig_rq4_per_cluster(eval_dir, out_dir / "rq4")
    fig_rq4_wabaq_moderation(eval_dir, out_dir / "rq4")

    #  Ablation figures 
    print("\nAblation figures:")
    fig_ablation_summary(eval_dir, out_dir / "ablation")
    fig_ablation_gru_effect(eval_dir, out_dir / "ablation")

    #  Sensitivity figures 
    print("\nSensitivity figures:")
    fig_sensitivity_overview(sens_dir, out_dir / "sensitivity")
    fig_sensitivity_stability_table(sens_dir, out_dir / "sensitivity")

    #  Overview / summary figures 
    print("\nOverview / summary figures:")
    fig_all_agents_metric_profile(eval_dir, out_dir / "overview")
    fig_ciu_improvement_distributions(eval_dir, out_dir / "overview")
    fig_pairwise_improvement_scatter(eval_dir, out_dir / "overview")
    fig_rq_summary_scorecard(eval_dir, out_dir / "overview")

    print(f"\n{'='*60}")
    print(f"Done. All figures saved to: {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA thesis plotting")
    parser.add_argument("--pes_dir",  default="outputs/pes")
    parser.add_argument("--rl_dir",   default="outputs/rl")
    parser.add_argument("--eval_dir", default="outputs/evaluation")
    parser.add_argument("--sens_dir", default="outputs/sensitivity")
    parser.add_argument("--out_dir",  default="outputs/evaluation/figures")
    args = parser.parse_args()
    main(args)