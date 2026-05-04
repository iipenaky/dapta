"""
Plots all 5 discourse metrics for RQ4:
DAPTA vs G-DDQN across pooled and per-cluster views.

Usage:
    python plot_discourse.py
    python plot_discourse.py --eval_dir outputs/evaluation --out_dir figures/rq4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# Global plot style applied to every figure in this script.
# Serif font and clean spines match the thesis formatting conventions.
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    # Remove top and right spines for a cleaner academic look.
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.5,
})

# Consistent colour scheme used across all RQ4 figures so the same agent
# always appears in the same colour regardless of which figure is shown.
AGENT_COLORS = {
    "DAPTA":  "#2166AC",   # Blue — personalised agent
    "G_DDQN": "#4DAC26",   # Green — generalised agent
    "RBDE":   "#636363",   # Grey — rule-based baseline
}

# Only CIU rate and Main Concept are the two primary discourse outcomes.
# These are the metrics with the strongest clinical validity evidence [Stark et al., 2021].
METRIC_LABELS = {
    "ciu_rate":             "CIU Rate",
    "mc_score":             "Main Concept",
}
METRIC_NAMES = list(METRIC_LABELS.keys())

# Minimum Cohen's d for a result to be considered clinically meaningful.
# Pre-specified as 0.40 following iTalkBetter [Upton et al., 2024].
CLINICAL_THRESHOLD = 0.40


def save(fig, out_dir: Path, name: str) -> None:
    """Saves a figure as both PDF (for the thesis) and PNG (for quick previewing).
    Creates the output directory if it does not already exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / f"{name}.pdf"))
    fig.savefig(str(out_dir / f"{name}.png"))
    plt.close(fig)
    print(f"  Saved: {name}")


def load_agent_arrays(arrays_dir: Path) -> dict:
    """Loads per-patient improvement arrays for every agent saved under arrays_dir.

    Each .npy file is expected to contain an (N, n_metrics) array where N is
    the number of test patients and each column is one discourse metric.
    The filename stem (e.g. 'test_DAPTA') is used as the agent key after
    stripping the 'test_' prefix.
    """
    agents = {}
    for f in sorted(arrays_dir.glob("*.npy")):
        name = f.stem.replace("test_", "")
        arr = np.load(str(f))
        # Skip empty files that may result from failed evaluation runs.
        if arr.size > 0:
            agents[name] = arr
    return agents


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cohen's d for paired samples (a minus b).

    Uses the standard deviation of the paired differences as the denominator,
    which is appropriate for within-patient comparisons across agents.
    Returns 0.0 if the standard deviation is zero (no variation in differences).
    """
    diff = a - b
    std = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 0 else 0.0


def load_json(path: Path) -> dict:
    """Loads a JSON file and returns its contents as a dictionary.
    Prints a warning and returns an empty dict if the file does not exist,
    allowing downstream functions to handle missing data gracefully.
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"  [WARN] Not found: {path}")
    return {}

#  Figure 1: Pooled — all 5 metrics side by side 
def fig_plot_discourse_pooled(eval_dir: Path, out_dir: Path):
    """
    5-panel figure: one subplot per metric.
    Each panel: boxplot of DAPTA vs G-DDQN with jittered points.
    Cohen's d and significance annotated on each panel.

    The paired lines connecting individual patients across the two agents
    make within-patient differences visible alongside the aggregate boxplots,
    so readers can see whether the pattern holds across patients or is driven
    by a few outliers.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents = load_agent_arrays(arrays_dir)
    dapta = agents.get("DAPTA")
    gddqn = agents.get("G_DDQN")
    if dapta is None or gddqn is None:
        print("  [WARN] DAPTA or G_DDQN arrays not found.")
        return

    # Use the smaller count to ensure arrays are the same length before pairing.
    n = min(len(dapta), len(gddqn))
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)

    for i, (metric, label) in enumerate(METRIC_LABELS.items()):
        ax = axes[i]
        d_vals = dapta[:n, i]
        g_vals = gddqn[:n, i]

        # Boxplots give the distributional summary (median, IQR, outliers).
        bx = ax.boxplot(
            [d_vals, g_vals],
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 2},
        )
        bx["boxes"][0].set_facecolor(AGENT_COLORS["DAPTA"])
        bx["boxes"][0].set_alpha(0.6)
        bx["boxes"][1].set_facecolor(AGENT_COLORS["G_DDQN"])
        bx["boxes"][1].set_alpha(0.6)

        # Jittered points show individual patients without overplotting.
        jitter = 0.10
        rng = np.random.default_rng(42)
        ax.scatter(
            np.ones(n) + rng.uniform(-jitter, jitter, n),
            d_vals, s=6, alpha=0.35, color=AGENT_COLORS["DAPTA"],
        )
        ax.scatter(
            np.ones(n) * 2 + rng.uniform(-jitter, jitter, n),
            g_vals, s=6, alpha=0.35, color=AGENT_COLORS["G_DDQN"],
        )

        # Paired lines connecting each patient's DAPTA and G-DDQN values.
        # Limited to 60 lines to keep the plot readable.
        n_lines = min(60, n)
        for j in range(n_lines):
            ax.plot(
                [1, 2], [d_vals[j], g_vals[j]],
                color="gray", alpha=0.08, linewidth=0.5,
            )

        # Compute Cohen's d and a Wilcoxon signed-rank p-value for annotation.
        # Wilcoxon is used because discourse improvements cannot be assumed normal.
        d = cohens_d(d_vals, g_vals)
        diff = d_vals - g_vals
        if np.any(diff != 0) and len(diff) >= 2:
            try:
                _, p = scipy_stats.wilcoxon(diff)
            except Exception:
                p = 1.0
        else:
            p = 1.0

        # Convert p-value to a star annotation and colour the title by clinical relevance.
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        color_d = "#2166AC" if abs(d) >= CLINICAL_THRESHOLD else "#888888"
        ax.set_title(f"{label}\nd={d:.2f} {star}", fontsize=10, color=color_d)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["DAPTA\n(pers.)", "G-DDQN\n(gen.)"], fontsize=9)
        # Only label the y-axis on the leftmost panel to avoid repetition.
        ax.set_ylabel("Improvement" if i == 0 else "")
        # Reference line at zero: above means improvement, below means decline.
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")

    fig.suptitle(
        "RQ4: All discourse metrics — DAPTA vs G-DDQN (test set, n={})".format(n),
        fontsize=13,
    )
    fig.tight_layout()
    save(fig, out_dir, "plot_discourse_pooled_boxplots")


#  Figure 2: Cohen's d for all metrics — DAPTA vs G-DDQN, forest plot ─
def fig_plot_discourse_forest(eval_dir: Path, out_dir: Path):
    """
    Forest plot: Cohen's d (DAPTA vs G-DDQN) for all 5 metrics.
    Positive d = personalisation wins, negative = generalisation wins.

    Bootstrap confidence intervals (1000 resamples) are computed for each
    metric so the plot conveys uncertainty alongside the point estimate.
    The two vertical reference lines mark zero (no difference) and the
    pre-specified clinical threshold (d = ±0.40).
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents = load_agent_arrays(arrays_dir)
    dapta = agents.get("DAPTA")
    gddqn = agents.get("G_DDQN")
    if dapta is None or gddqn is None:
        return

    n = min(len(dapta), len(gddqn))
    ds, cis, stars = [], [], []

    for i in range(5):
        d_vals = dapta[:n, i]
        g_vals = gddqn[:n, i]
        d = cohens_d(d_vals, g_vals)
        diff = d_vals - g_vals

        # Bootstrap 95% CI by resampling the paired differences 1000 times.
        lo = float(np.percentile(
            [np.mean(np.random.choice(diff, len(diff), replace=True)) for _ in range(1000)],
            2.5,
        ))
        hi = float(np.percentile(
            [np.mean(np.random.choice(diff, len(diff), replace=True)) for _ in range(1000)],
            97.5,
        ))
        if np.any(diff != 0) and len(diff) >= 2:
            try:
                _, p = scipy_stats.wilcoxon(diff)
            except Exception:
                p = 1.0
        else:
            p = 1.0
        ds.append(d)
        cis.append((lo, hi))
        stars.append("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")

    fig, ax = plt.subplots(figsize=(7, 5))
    # y positions for each metric row in the forest plot.
    y = np.arange(5)

    for i, (d, (lo, hi), star) in enumerate(zip(ds, cis, stars)):
        # Colour the point by which agent wins for this metric.
        color = AGENT_COLORS["DAPTA"] if d > 0 else AGENT_COLORS["G_DDQN"]
        # Clamp error bar extents to zero to avoid negative widths from rounding.
        xerr_lo = max(d - lo, 0.0)
        xerr_hi = max(hi - d, 0.0)
        ax.errorbar(
            d, y[i],
            xerr=[[xerr_lo], [xerr_hi]],
            fmt="o", color=color, capsize=5,
            markersize=7, elinewidth=1.5, capthick=1.5,
        )
        # Place the significance star just to the right of the upper CI bound.
        ax.text(
            hi + 0.01, y[i], star,
            va="center", fontsize=10, color=color,
        )

    # Reference line at d = 0: no difference between personalised and generalised.
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    # Clinical threshold lines at ±0.40 to visually flag meaningful effect sizes.
    ax.axvline(CLINICAL_THRESHOLD, color="#D73027", linewidth=0.8,
               linestyle=":", alpha=0.7, label=f"Clinical threshold (d={CLINICAL_THRESHOLD})")
    ax.axvline(-CLINICAL_THRESHOLD, color="#D73027", linewidth=0.8,
               linestyle=":", alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(list(METRIC_LABELS.values()))
    ax.set_xlabel("Cohen's d  (DAPTA − G-DDQN)\nBlue = personalisation wins  |  Green = generalisation wins")
    ax.set_title("RQ4: Personalisation benefit across all discourse metrics")
    ax.legend(fontsize=9)

    # Light background shading to distinguish which side favours each agent.
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], 0, alpha=0.04, color=AGENT_COLORS["G_DDQN"])
    ax.axvspan(0, xlim[1], alpha=0.04, color=AGENT_COLORS["DAPTA"])
    ax.set_xlim(xlim)

    fig.tight_layout()
    save(fig, out_dir, "plot_discourse_forest")


#  Figure 3: Per-cluster heatmap — all 5 metrics 
def fig_rq4_per_cluster_all_metrics(eval_dir: Path, out_dir: Path):
    """
    Heatmap: rows = clusters, columns = 5 metrics.
    Cell value = Cohen's d (DAPTA − G-DDQN).
    Green = personalisation wins, red = generalisation wins.

    This figure directly supports the RQ4 cluster-level heterogeneity finding:
    it makes visible which aphasia subtypes benefit from personalisation and
    which are better served by the generalised agent.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents = load_agent_arrays(arrays_dir)
    dapta = agents.get("DAPTA")
    gddqn = agents.get("G_DDQN")
    rq4 = load_json(eval_dir / "test_rq4_personalisation.json")

    if dapta is None or gddqn is None:
        return

    # Get cluster assignments from cluster_performance or rq4
    per_cluster = rq4.get("per_cluster", {})
    if not per_cluster:
        print("  [WARN] No per_cluster data in test_rq4_personalisation.json")
        return

    # Load the cluster labels saved during PES training.
    # split_labels identifies which patients belong to the test set.
    pes_data = np.load("outputs/pes/env_initial_states.npz", allow_pickle=True)
    split_labels = pes_data["split_labels"]
    cluster_labels = pes_data["cluster_labels"]
    test_mask = split_labels == "test"
    predicted_clusters = cluster_labels[test_mask]

    n = min(len(dapta), len(gddqn), len(predicted_clusters))
    # Sort clusters numerically so rows appear in a consistent order.
    clusters = sorted(per_cluster.keys(), key=int)

    # Build the (n_clusters × n_metrics) Cohen's d matrix.
    matrix = np.zeros((len(clusters), 5))
    for ci, c in enumerate(clusters):
        mask = predicted_clusters[:n] == int(c)
        # Skip clusters with fewer than 2 patients — d is undefined.
        if mask.sum() < 2:
            continue
        for mi in range(2):
            d_vals = dapta[:n][mask, mi]
            g_vals = gddqn[:n][mask, mi]
            matrix[ci, mi] = cohens_d(d_vals, g_vals)

    fig, ax = plt.subplots(figsize=(10, max(3, len(clusters) * 0.8)))
    # Symmetric colour scale centred on zero so equal magnitudes appear equally vivid.
    vmax = max(abs(matrix).max(), 0.5)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(2))
    ax.set_xticklabels(list(METRIC_LABELS.values()), rotation=20, ha="right")
    ax.set_yticks(range(len(clusters)))

    # Build row labels that include the dominant aphasia subtype and cluster size
    # so readers can interpret the heatmap without consulting a separate table.
    row_labels = []
    for c in clusters:
        dom = per_cluster[c].get("dominant_subtype", "?")
        n_c = per_cluster[c].get("n", "?")
        row_labels.append(f"C{c} — {dom} (n={n_c})")
    ax.set_yticklabels(row_labels)

    # Annotate each cell with the numeric d value.
    # Switch text colour to white when the background is dark enough to cause contrast issues.
    for ci in range(len(clusters)):
        for mi in range(2):
            val = matrix[ci, mi]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(mi, ci, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax, label="Cohen's d (DAPTA − G-DDQN)", shrink=0.6)
    ax.set_title(
        "RQ4: Per-cluster personalisation benefit across all discourse metrics\n"
        "(green = personalisation wins, red = generalisation wins)"
    )
    fig.tight_layout()
    save(fig, out_dir, "rq4_per_cluster_all_metrics_heatmap")

#  Figure 4: Per-cluster bar charts — all 5 metrics ─
def fig_rq4_per_cluster_all_metrics_bars(eval_dir: Path, out_dir: Path):
    """
    5 subplots (one per metric), each showing DAPTA vs G-DDQN per cluster.

    Bar charts show absolute mean improvement rather than Cohen's d, making
    the magnitude of change directly comparable to the clinical literature
    and to the pooled results reported in Table 4.4 of the thesis.
    """
    arrays_dir = eval_dir / "test_per_agent_arrays"
    agents = load_agent_arrays(arrays_dir)
    dapta = agents.get("DAPTA")
    gddqn = agents.get("G_DDQN")
    rq4 = load_json(eval_dir / "test_rq4_personalisation.json")

    if dapta is None or gddqn is None:
        return

    per_cluster = rq4.get("per_cluster", {})
    if not per_cluster:
        return

    # Load cluster assignments for test patients, same as in the heatmap function.
    pes_data = np.load("outputs/pes/env_initial_states.npz", allow_pickle=True)
    split_labels = pes_data["split_labels"]
    cluster_labels = pes_data["cluster_labels"]
    test_mask = split_labels == "test"
    predicted_clusters = cluster_labels[test_mask]

    n = min(len(dapta), len(gddqn), len(predicted_clusters))
    clusters = sorted(per_cluster.keys(), key=int)
    # x positions for cluster groups; bars are offset by half a width each side.
    x = np.arange(len(clusters))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for mi, (metric, label) in enumerate(METRIC_LABELS.items()):
        ax = axes[mi]
        d_means, g_means = [], []

        for c in clusters:
            mask = predicted_clusters[:n] == int(c)
            # Fall back to 0.0 for clusters with no test patients.
            d_means.append(float(np.mean(dapta[:n][mask, mi])) if mask.sum() > 0 else 0.0)
            g_means.append(float(np.mean(gddqn[:n][mask, mi])) if mask.sum() > 0 else 0.0)

        bars_d = ax.bar(x - width / 2, d_means, width,
                        color=AGENT_COLORS["DAPTA"], alpha=0.8,
                        label="DAPTA (pers.)", edgecolor="white")
        bars_g = ax.bar(x + width / 2, g_means, width,
                        color=AGENT_COLORS["G_DDQN"], alpha=0.8,
                        label="G-DDQN (gen.)", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in clusters])
        ax.set_title(label, fontsize=11)
        # Only label the y-axis on the leftmost panel to avoid repetition.
        ax.set_ylabel("Mean improvement" if mi == 0 else "")
        # Reference line at zero: bars above mean improvement, below mean decline.
        ax.axhline(0, color="gray", linewidth=0.6)

        # Only add the legend to the first panel to avoid duplication.
        if mi == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "RQ4: Per-cluster results across all discourse metrics — DAPTA vs G-DDQN",
        fontsize=13,
    )
    fig.tight_layout()
    save(fig, out_dir, "rq4_per_cluster_all_metrics_bars")



# Main 
def main(args):
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)

    print("\nGenerating RQ4 all-metrics figures...\n")

    # Figures 3 and 4 are the cluster-level analyses most relevant to the
    # RQ4 heterogeneity finding; Figures 1 and 2 provide the pooled overview.
    fig_rq4_per_cluster_all_metrics(eval_dir, out_dir)
    fig_rq4_per_cluster_all_metrics_bars(eval_dir, out_dir)

    print(f"\nDone. Figures saved to: {out_dir}/")
    print("  plot_discourse_pooled_boxplots")
    print("  plot_discourse_forest")
    print("  rq4_per_cluster_all_metrics_heatmap")
    print("  rq4_per_cluster_all_metrics_bars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4 all metrics plots")
    parser.add_argument("--eval_dir", default="outputs/evaluation")
    parser.add_argument("--out_dir",  default="outputs/evaluation/figures/rq4")
    args = parser.parse_args()
    main(args)